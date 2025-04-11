from flask import Flask, request, jsonify, render_template
import requests
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import hashlib
from collections import Counter

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Check if we should use simple embeddings
use_simple_embeddings = os.getenv('USE_SIMPLE_EMBEDDINGS', 'false').lower() == 'true'

# Initialize the embedding model and vector database
if not use_simple_embeddings:
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        
        # Initialize the embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create a vector database
        embedding_size = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(embedding_size)
    except ImportError as e:
        print(f"Warning: Could not import advanced embedding libraries: {e}")
        print("Falling back to simple embeddings")
        use_simple_embeddings = True

# Initialize TinyLlama model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    
    # Load TinyLlama model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading TinyLlama model: {model_name}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with lower precision for efficiency
    tinyllama_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tinyllama_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    # Create a text generation pipeline
    tinyllama_pipeline = pipeline(
        "text-generation",
        model=tinyllama_model,
        tokenizer=tinyllama_tokenizer,
        # Remove max_length parameter here to avoid the conflict
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        device=device if device == "cuda" else -1
    )
    
    use_tinyllama = True
    print("TinyLlama model loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import TinyLlama dependencies: {e}")
    print("Falling back to rule-based recipe generation")
    use_tinyllama = False
except Exception as e:
    print(f"Error loading TinyLlama model: {e}")
    print("Falling back to rule-based recipe generation")
    use_tinyllama = False

# Simple embedding database for fallback
recipe_data = []
recipe_vectors = []

# Simple embedding function using word frequencies
def simple_text_embedding(text, dim=100):
    # Convert text to lowercase and split into words
    words = text.lower().split()
    # Count word frequencies
    word_counts = Counter(words)
    # Create a hash for each word and use it to determine vector position
    vector = np.zeros(dim)
    for word, count in word_counts.items():
        # Create a hash of the word
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        # Use the hash to determine the position in the vector
        pos = hash_val % dim
        # Add the count to that position
        vector[pos] += count
    # Normalize the vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def simple_vector_search(query_vector, vectors, top_k=5):
    similarities = []
    for i, vec in enumerate(vectors):
        similarity = np.dot(query_vector, vec)
        similarities.append((i, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Return top k indices
    return [idx for idx, _ in similarities[:top_k]]

# Function to fetch recipes from API
def fetch_recipes(query):
    api_choice = os.getenv('USE_API', 'themealdb').lower()
    
    if api_choice == 'themealdb':
        # TheMealDB API - free, no authentication required
        # First try with the provided query
        url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Check if meals is None or empty
            if not data.get('meals'):
                # If no results, try searching by first letter
                first_letter = query[0] if query else 'c'
                url = f"https://www.themealdb.com/api/json/v1/1/search.php?f={first_letter}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                
                # If still no results, get random meals
                if not data.get('meals'):
                    # Get some random meals as fallback
                    random_meals = []
                    for _ in range(5):  # Try to get 5 random meals
                        random_url = "https://www.themealdb.com/api/json/v1/1/random.php"
                        random_response = requests.get(random_url)
                        if random_response.status_code == 200:
                            random_data = random_response.json()
                            if random_data.get('meals'):
                                random_meals.extend(random_data.get('meals'))
                    
                    if random_meals:
                        data = {'meals': random_meals}
            
            # Convert TheMealDB format to our standard format
            result = {"hits": []}
            
            # Now safely iterate over meals (which might still be None)
            for meal in data.get('meals', []):
                # Extract ingredients (TheMealDB has ingredients in separate fields)
                ingredients = []
                for i in range(1, 21):  # TheMealDB has up to 20 ingredients
                    ingredient = meal.get(f'strIngredient{i}')
                    measure = meal.get(f'strMeasure{i}')
                    if ingredient and ingredient.strip():
                        ingredients.append(f"{measure} {ingredient}".strip())
                
                result["hits"].append({
                    "recipe": {
                        "label": meal.get('strMeal', ''),
                        "ingredientLines": ingredients,
                        "url": meal.get('strSource', ''),
                        "image": meal.get('strMealThumb', ''),
                        "source": "TheMealDB",
                        "calories": 0,  # Not provided by TheMealDB
                        "instructions": meal.get('strInstructions', '')
                    }
                })
            
            return result
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
    
    elif api_choice == 'spoonacular':
        # Spoonacular API - requires API key but has free tier
        api_key = os.getenv('SPOONACULAR_API_KEY')
        if not api_key:
            return {"error": "Spoonacular API key not found in .env file"}
        
        url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&apiKey={api_key}&addRecipeInformation=true&number=10"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            result = {"hits": []}
            
            for recipe in data.get('results', []):
                result["hits"].append({
                    "recipe": {
                        "label": recipe.get('title', ''),
                        "ingredientLines": [ingredient.get('original', '') for ingredient in recipe.get('extendedIngredients', [])],
                        "url": recipe.get('sourceUrl', ''),
                        "image": recipe.get('image', ''),
                        "source": recipe.get('sourceName', 'Spoonacular'),
                        "calories": recipe.get('nutrition', {}).get('nutrients', [{}])[0].get('amount', 0) if 'nutrition' in recipe else 0,
                        "instructions": recipe.get('instructions', '')
                    }
                })
            
            return result
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
    
    else:
        # Original Edamam API
        app_id = os.getenv('EDAMAM_APP_ID')
        app_key = os.getenv('EDAMAM_APP_KEY')
        
        if not app_id or not app_key:
            return {"error": "Edamam API credentials not found in .env file"}
        
        url = f"https://api.edamam.com/search?q={query}&app_id={app_id}&app_key={app_key}"
        
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status code {response.status_code}"}

# Function to index recipes
def index_recipes(recipes):
    global recipe_data, recipe_vectors
    
    # Clear existing data
    recipe_data = []
    recipe_vectors = []
    
    if not use_simple_embeddings:
        global index
        if index.ntotal > 0:
            index = faiss.IndexFlatL2(embedding_size)
    
    # Process and index each recipe
    for recipe in recipes.get('hits', []):
        recipe_info = recipe.get('recipe', {})
        recipe_text = f"{recipe_info.get('label', '')} - {', '.join(recipe_info.get('ingredientLines', []))}"
        
        # Create embedding
        if use_simple_embeddings:
            embedding = simple_text_embedding(recipe_text)
            recipe_vectors.append(embedding)
        else:
            embedding = model.encode([recipe_text])[0]
            # Add to index
            index.add(np.array([embedding], dtype=np.float32))
        
        # Store recipe data
        recipe_data.append({
            'title': recipe_info.get('label', ''),
            'ingredients': recipe_info.get('ingredientLines', []),
            'url': recipe_info.get('url', ''),
            'image': recipe_info.get('image', ''),
            'source': recipe_info.get('source', ''),
            'calories': recipe_info.get('calories', 0),
            'instructions': recipe_info.get('instructions', ''),
            'text': recipe_text
        })
    
    return len(recipe_data)

# Function to search for similar recipes
def search_similar_recipes(query, top_k=5):
    # Create embedding for the query
    if use_simple_embeddings:
        query_embedding = simple_text_embedding(query)
        
        # Search in the vectors
        if recipe_vectors:
            indices = simple_vector_search(query_embedding, recipe_vectors, top_k)
            
            # Get the results
            results = []
            for idx in indices:
                if idx < len(recipe_data):
                    results.append(recipe_data[idx])
            
            return results
        return []
    else:
        query_embedding = model.encode([query])[0]
        
        # Search in the index
        distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
        
        # Get the results
        results = []
        for idx in indices[0]:
            if idx < len(recipe_data):
                results.append(recipe_data[idx])
        
        return results

# Function to generate recipe using TinyLlama
def generate_recipe_with_tinyllama(query, similar_recipes, preferences):
    # Create a prompt for TinyLlama
    prompt = f"<human>: Create a recipe for {query}.\n\n"
    
    # Add dietary preferences to the prompt
    if preferences.get('vegetarian'):
        prompt += "The recipe should be vegetarian.\n"
    if preferences.get('vegan'):
        prompt += "The recipe should be vegan.\n"
    if preferences.get('glutenFree'):
        prompt += "The recipe should be gluten-free.\n"
    if preferences.get('lowCarb'):
        prompt += "The recipe should be low-carb.\n"
    
    # Add cooking time preference
    cooking_time = "medium"
    if preferences.get('cookingTime'):
        cooking_time = preferences.get('cookingTime')
    prompt += f"The cooking time should be {cooking_time}.\n\n"
    
    # Add information from similar recipes
    prompt += "Here are some similar recipes for inspiration:\n"
    for i, recipe in enumerate(similar_recipes[:3], 1):
        prompt += f"Recipe {i}: {recipe['title']}\n"
        prompt += f"Ingredients: {', '.join(recipe['ingredients'][:5])}\n"
        if recipe.get('instructions'):
            # Add a short excerpt of instructions
            instructions_excerpt = recipe['instructions'][:200] + "..."
            prompt += f"Instructions excerpt: {instructions_excerpt}\n"
        prompt += "\n"
    
    prompt += "Please create a new recipe with a title, ingredients list, and step-by-step instructions.\n"
    prompt += "</human>\n\n<assistant>:"
    
    # Generate text with TinyLlama
    try:
        result = tinyllama_pipeline(
            prompt, 
            max_new_tokens=800, 
            do_sample=True, 
            temperature=0.7
        )[0]['generated_text']
        
        # Extract the assistant's response
        response = result.split("<assistant>:")[1].strip()
        
        # Parse the generated recipe
        title = f"AI-Generated Recipe for {query}"
        ingredients = []
        instructions = ""
        
        # Try to extract title, ingredients, and instructions
        lines = response.split('\n')
        section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for title
            if "title:" in line.lower() or "recipe for" in line.lower():
                title = line
                continue
                
            # Check for section headers
            if "ingredients:" in line.lower():
                section = "ingredients"
                continue
            elif "instructions:" in line.lower() or "directions:" in line.lower() or "steps:" in line.lower():
                section = "instructions"
                instructions += line + "\n"
                continue
                
            # Add content to the appropriate section
            if section == "ingredients":
                # Check if line starts with a number or bullet point
                if line.startswith(("-", "*", "•")) or (line[0].isdigit() and "." in line[:3]):
                    ingredients.append(line.lstrip("- *•0123456789. "))
                else:
                    ingredients.append(line)
            elif section == "instructions":
                instructions += line + "\n"
        
        # If we couldn't parse ingredients properly, try another approach
        if not ingredients:
            # Look for lines that might be ingredients (contain measurements)
            for line in lines:
                line = line.strip()
                if any(unit in line.lower() for unit in ["cup", "tbsp", "tsp", "tablespoon", "teaspoon", "ounce", "oz", "pound", "lb", "gram", "g", "ml", "liter"]):
                    ingredients.append(line)
        
        # If still no ingredients, use the similar recipes
        if not ingredients and similar_recipes:
            all_ingredients = []
            instructions = []
            cooking_steps = []
            
            for recipe in similar_recipes:
                all_ingredients.extend(recipe['ingredients'])
                if recipe.get('instructions'):
                    instructions.append(recipe.get('instructions'))
            
            # Get unique ingredients
            unique_ingredients = list(set(all_ingredients))
            
            # Limit to a reasonable number of ingredients
            if len(unique_ingredients) > 15:
                unique_ingredients = unique_ingredients[:15]
            
            # Generate cooking steps
            if instructions:
                # Extract sentences from instructions
                for instruction in instructions:
                    sentences = instruction.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 10:  # Only include meaningful sentences
                            cooking_steps.append(sentence.strip() + '.')
                    
                # Limit to a reasonable number of steps
                if len(cooking_steps) > 10:
                    cooking_steps = cooking_steps[:10]
            
            # Get an image from one of the similar recipes
            image_url = ""
            for recipe in similar_recipes:
                if recipe.get('image'):
                    image_url = recipe.get('image')
                    break
            
            # Generate cooking time based on preference
            cooking_time = "30-45 minutes"
            if preferences.get('cookingTime') == 'quick':
                cooking_time = "15-25 minutes"
            elif preferences.get('cookingTime') == 'medium':
                cooking_time = "30-45 minutes"
            elif preferences.get('cookingTime') == 'slow':
                cooking_time = "60-90 minutes"
            
            # Create a new recipe
            generated_recipe = {
                'title': title,
                'ingredients': ingredients,
                'instructions': instructions,
                'cooking_time': cooking_time_text,
                'image': image_url,
                'similar_recipes': similar_recipes[:3]  # Include top 3 similar recipes for reference
            }
            
            return generated_recipe
    except Exception as e:
            print(f"Error generating recipe with TinyLlama: {e}")
            return None

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    query = data.get('query', '')
    preferences = data.get('preferences', {})
    
    # Modify query based on preferences
    modified_query = query
    if preferences.get('vegetarian'):
        modified_query += ' vegetarian'
    if preferences.get('vegan'):
        modified_query += ' vegan'
    if preferences.get('glutenFree'):
        modified_query += ' gluten-free'
    if preferences.get('lowCarb'):
        modified_query += ' low carb'
    
    if len(recipe_data) == 0:
        # If no recipes are indexed, fetch some first
        recipes = fetch_recipes('chicken')  # Changed from 'popular' to 'chicken' which is more likely to return results
        
        # If still no recipes, try with other common ingredients
        if not recipes.get('hits'):
            for fallback_query in ['beef', 'pasta', 'vegetable', 'dessert']:
                recipes = fetch_recipes(fallback_query)
                if recipes.get('hits'):
                    break
        
        index_recipes(recipes)
    
    # Get similar recipes
    similar_recipes = search_similar_recipes(modified_query)
    
    # Generate a new recipe based on the retrieved recipes
    if similar_recipes:
        # Try to use TinyLlama if available
        if use_tinyllama:
            generated_recipe = generate_recipe_with_tinyllama(query, similar_recipes, preferences)
            if generated_recipe:
                return jsonify(generated_recipe)
        
        # Fall back to rule-based generation if TinyLlama fails or is not available
        # Simple approach: take ingredients from similar recipes and combine them
        all_ingredients = []
        instructions = []
        cooking_steps = []
        
        for recipe in similar_recipes:
            all_ingredients.extend(recipe['ingredients'])
            if recipe.get('instructions'):
                instructions.append(recipe.get('instructions'))
        
        # Get unique ingredients
        unique_ingredients = list(set(all_ingredients))
        
        # Limit to a reasonable number of ingredients
        if len(unique_ingredients) > 15:
            unique_ingredients = unique_ingredients[:15]
        
        # Generate cooking steps
        if instructions:
            # Extract sentences from instructions
            for instruction in instructions:
                sentences = instruction.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Only include meaningful sentences
                        cooking_steps.append(sentence.strip() + '.')
            
            # Limit to a reasonable number of steps
            if len(cooking_steps) > 10:
                cooking_steps = cooking_steps[:10]
        
            # Get an image from one of the similar recipes
            image_url = ""
            for recipe in similar_recipes:
                if recipe.get('image'):
                    image_url = recipe.get('image')
                    break
            
            # Generate cooking time based on preference
            cooking_time = "30-45 minutes"
            if preferences.get('cookingTime') == 'quick':
                cooking_time = "15-25 minutes"
            elif preferences.get('cookingTime') == 'medium':
                cooking_time = "30-45 minutes"
            elif preferences.get('cookingTime') == 'slow':
                cooking_time = "60-90 minutes"
            
            # Create a new recipe
            generated_recipe = {
                'title': f"AI-Generated Recipe for {query}",
                'ingredients': unique_ingredients,
                'instructions': ' '.join(cooking_steps),
                'cooking_time': cooking_time,
                'image': image_url,
                'similar_recipes': similar_recipes[:3]  # Include top 3 similar recipes for reference
            }
            
            return jsonify(generated_recipe)
        
        # If no similar recipes found, return an error
        return jsonify({"error": "Could not find similar recipes to generate from"})

@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    data = request.json
    query = data.get('query', '')
    
    recipes = fetch_recipes(query)
    index_recipes(recipes)
    
    return jsonify(recipes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/similar', methods=['POST'])
def api_similar():
    data = request.json
    query = data.get('query', '')
    
    similar_recipes = search_similar_recipes(query)
    
    return jsonify(similar_recipes)

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

# Run the app
if __name__ == '__main__':
    # Run the app on all network interfaces (0.0.0.0) and port 8080
    # This makes it accessible from other devices on your network
    app.run(host='0.0.0.0', port=8080, debug=True)
        