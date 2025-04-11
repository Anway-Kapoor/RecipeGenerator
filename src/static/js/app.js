// Store saved recipes and history
let savedRecipes = JSON.parse(localStorage.getItem('savedRecipes') || '[]');
let recipeHistory = JSON.parse(localStorage.getItem('recipeHistory') || '[]');

// Show tab content
function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Add active class to clicked button
    Array.from(document.querySelectorAll('.tab-button')).find(
        button => button.getAttribute('onclick').includes(tabId)
    ).classList.add('active');
    
    // Load content for the tab
    if (tabId === 'saved-recipes') {
        loadSavedRecipes();
    } else if (tabId === 'recipe-history') {
        loadRecipeHistory();
    }
}

// Load saved recipes
function loadSavedRecipes() {
    const container = document.getElementById('saved-recipes-container');
    
    if (savedRecipes.length === 0) {
        container.innerHTML = '<p>You haven\'t saved any recipes yet.</p>';
        return;
    }
    
    let html = '<div class="recipe-grid">';
    
    savedRecipes.forEach((recipe, index) => {
        html += `
            <div class="recipe-grid-item">
                ${recipe.image ? `<img src="${recipe.image}" alt="${recipe.title}" class="recipe-grid-image">` : ''}
                <div class="recipe-grid-content">
                    <h3 class="recipe-grid-title">${recipe.title}</h3>
                    <p class="recipe-grid-meta">${recipe.ingredients.length} ingredients</p>
                    <button class="recipe-grid-button" onclick="viewSavedRecipe(${index})">View Recipe</button>
                    <button class="recipe-grid-button" style="background-color: #f44336;" onclick="deleteSavedRecipe(${index})">Delete</button>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// Load recipe history
function loadRecipeHistory() {
    const container = document.getElementById('history-container');
    
    if (recipeHistory.length === 0) {
        container.innerHTML = '<p>No recipe generation history yet.</p>';
        return;
    }
    
    let html = '<div class="recipe-grid">';
    
    recipeHistory.forEach((recipe, index) => {
        html += `
            <div class="recipe-grid-item">
                ${recipe.image ? `<img src="${recipe.image}" alt="${recipe.title}" class="recipe-grid-image">` : ''}
                <div class="recipe-grid-content">
                    <h3 class="recipe-grid-title">${recipe.title}</h3>
                    <p class="recipe-grid-meta">Generated on ${recipe.date}</p>
                    <button class="recipe-grid-button" onclick="viewHistoryRecipe(${index})">View Recipe</button>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// View a saved recipe
function viewSavedRecipe(index) {
    const recipe = savedRecipes[index];
    displayRecipe(recipe);
    showTab('recipe-result');
}

// View a history recipe
function viewHistoryRecipe(index) {
    const recipe = recipeHistory[index];
    displayRecipe(recipe);
    showTab('recipe-result');
}

// Delete a saved recipe
function deleteSavedRecipe(index) {
    if (confirm('Are you sure you want to delete this recipe?')) {
        savedRecipes.splice(index, 1);
        localStorage.setItem('savedRecipes', JSON.stringify(savedRecipes));
        loadSavedRecipes();
    }
}

// Display a recipe in the main view
function displayRecipe(recipe) {
    const resultDiv = document.getElementById('recipe-result');
    
    // Generate random nutrition facts for demo purposes
    const calories = Math.floor(Math.random() * 500) + 200;
    const protein = Math.floor(Math.random() * 30) + 10;
    const carbs = Math.floor(Math.random() * 50) + 20;
    const fat = Math.floor(Math.random() * 20) + 5;
    
    let html = `
        <div class="recipe recipe-card">
            ${recipe.image ? `<img src="${recipe.image}" alt="${recipe.title}" class="recipe-image">` : ''}
            <div style="padding: 20px;">
                <h2>${recipe.title}</h2>
                
                <div class="nutrition-info">
                    <h3>Estimated Nutrition Facts</h3>
                    <div class="nutrition-chart">
                        <div class="nutrition-item">
                            <div class="nutrition-value">${calories}</div>
                            <div class="nutrition-label">Calories</div>
                        </div>
                        <div class="nutrition-item">
                            <div class="nutrition-value">${protein}g</div>
                            <div class="nutrition-label">Protein</div>
                        </div>
                        <div class="nutrition-item">
                            <div class="nutrition-value">${carbs}g</div>
                            <div class="nutrition-label">Carbs</div>
                        </div>
                        <div class="nutrition-item">
                            <div class="nutrition-value">${fat}g</div>
                            <div class="nutrition-label">Fat</div>
                        </div>
                    </div>
                </div>
                
                <div class="ingredients">
                    <h3>Ingredients:</h3>
                    <ul>
    `;
    
    recipe.ingredients.forEach(ingredient => {
        html += `<li>${ingredient}</li>`;
    });
    
    html += `
                    </ul>
                </div>
                <div class="instructions">
                    <h3>Instructions:</h3>
                    <p>${recipe.instructions}</p>
                </div>
                
                <div class="recipe-actions">
                    <button class="action-button save-button" onclick="saveRecipe(${JSON.stringify(recipe).replace(/"/g, '&quot;')})">Save Recipe</button>
                    <button class="action-button print-button" onclick="printRecipe()">Print Recipe</button>
                    <button class="action-button" style="background-color: #FF9800; color: white;" onclick="shareRecipe()">Share Recipe</button>
                </div>
            `;
            
            if (recipe.similar_recipes && recipe.similar_recipes.length > 0) {
                html += `
                    <div class="similar-recipes">
                        <h3>Similar Recipes You Might Like:</h3>
                        <div style="display: flex; gap: 15px; overflow-x: auto; padding: 10px 0;">
                `;
                
                recipe.similar_recipes.forEach(similar => {
                    html += `
                        <div class="similar-recipe" style="min-width: 200px; max-width: 200px;">
                            ${similar.image ? `<img src="${similar.image}" alt="${similar.title}" style="width: 100%; height: 120px; object-fit: cover; border-radius: 5px;">` : ''}
                            <h4>${similar.title}</h4>
                            <p><strong>Source:</strong> ${similar.source}</p>
                            <p><a href="${similar.url}" target="_blank">View Original Recipe</a></p>
                        </div>
                    `;
                });
                
                html += `</div></div>`;
            }
            
            html += `</div></div>`;
            resultDiv.innerHTML = html;
}

async function fetchRecipes() {
    const query = document.getElementById('query').value;
    if (!query) return;
    
    // Get dietary preferences
    const preferences = {
        vegetarian: document.getElementById('vegetarian').checked,
        vegan: document.getElementById('vegan').checked,
        glutenFree: document.getElementById('gluten-free').checked,
        lowCarb: document.getElementById('low-carb').checked,
        cookingTime: document.getElementById('cooking-time').value
    };
    
    try {
        const response = await fetch('/api/fetch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                query,
                preferences
            })
        });
        
        const data = await response.json();
        console.log('Fetched recipes:', data);
        return data;
    } catch (error) {
        console.error('Error fetching recipes:', error);
        return null;
    }
}

async function generateRecipe() {
    const query = document.getElementById('query').value;
    if (!query) {
        alert('Please enter ingredients or a dish name');
        return;
    }
    
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('recipe-result');
    
    loadingDiv.style.display = 'block';
    resultDiv.innerHTML = '';
    showTab('recipe-result');
    
    try {
        // Get dietary preferences
        const preferences = {
            vegetarian: document.getElementById('vegetarian').checked,
            vegan: document.getElementById('vegan').checked,
            glutenFree: document.getElementById('gluten-free').checked,
            lowCarb: document.getElementById('low-carb').checked,
            cookingTime: document.getElementById('cooking-time').value
        };
        
        // First fetch recipes to ensure we have data
        await fetchRecipes();
        
        // Then generate a recipe
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                query,
                preferences
            })
        });
        
        const recipe = await response.json();
        
        if (recipe.error) {
            resultDiv.innerHTML = `<div class="recipe"><h2>Error</h2><p>${recipe.error}</p></div>`;
            
            if (recipe.suggestions) {
                let suggestionsHtml = `<p>Try one of these suggestions:</p><div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">`;
                recipe.suggestions.forEach(suggestion => {
                    suggestionsHtml += `<button onclick="document.getElementById('query').value='${suggestion}'; generateRecipe()" style="padding: 8px 12px; background: #f1f1f1; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;">${suggestion}</button>`;
                });
                suggestionsHtml += `</div>`;
                resultDiv.innerHTML += suggestionsHtml;
            }
        } else {
            // Add to history
            const now = new Date();
            const historyItem = {
                ...recipe,
                date: now.toLocaleDateString() + ' ' + now.toLocaleTimeString()
            };
            
            // Add to the beginning of the array
            recipeHistory.unshift(historyItem);
            
            // Keep only the last 10 items
            if (recipeHistory.length > 10) {
                recipeHistory = recipeHistory.slice(0, 10);
            }
            
            // Save to localStorage
            localStorage.setItem('recipeHistory', JSON.stringify(recipeHistory));
            
            // Display the recipe
            displayRecipe(recipe);
        }
    } catch (error) {
        console.error('Error generating recipe:', error);
        resultDiv.innerHTML = `<div class="recipe"><h2>Error</h2><p>Failed to generate recipe. Please try again.</p></div>`;
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function saveRecipe(recipe) {
    // Check if recipe already exists in saved recipes
    const exists = savedRecipes.some(r => r.title === recipe.title);
    
    if (exists) {
        alert('This recipe is already saved!');
        return;
    }
    
    // Add timestamp
    recipe.savedAt = new Date().toLocaleString();
    
    // Add to the beginning of saved recipes
    savedRecipes.unshift(recipe);
    
    // Save to localStorage
    localStorage.setItem('savedRecipes', JSON.stringify(savedRecipes));
    
    // Show confirmation
    alert('Recipe saved successfully!');
}

function printRecipe() {
    // Create a print-friendly version
    const content = document.getElementById('recipe-result').innerHTML;
    const printWindow = window.open('', '_blank');
    
    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recipe Print</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .recipe-image { max-width: 100%; height: auto; }
                .recipe-actions, .similar-recipes { display: none; }
                .nutrition-info { background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 15px 0; }
                .nutrition-chart { display: flex; justify-content: space-between; }
                .nutrition-item { text-align: center; flex: 1; }
                .nutrition-value { font-size: 18px; font-weight: bold; }
                .nutrition-label { font-size: 12px; color: #666; }
                h2 { color: #2e7d32; }
                h3 { color: #2e7d32; margin-top: 20px; }
                ul { padding-left: 20px; }
                li { margin-bottom: 5px; }
                @media print {
                    .no-print { display: none; }
                    body { font-size: 12pt; }
                    h2 { font-size: 16pt; }
                    h3 { font-size: 14pt; }
                }
            </style>
        </head>
        <body>
            <div class="no-print" style="margin-bottom: 20px;">
                <button onclick="window.print()">Print Recipe</button>
                <button onclick="window.close()">Close</button>
            </div>
            ${content}
        </body>
        </html>
    `);
    
    printWindow.document.close();
}

function shareRecipe() {
    // Check if Web Share API is supported
    if (navigator.share) {
        const recipeTitle = document.querySelector('.recipe h2').textContent;
        const recipeUrl = window.location.href;
        
        navigator.share({
            title: recipeTitle,
            text: `Check out this recipe: ${recipeTitle}`,
            url: recipeUrl
        })
        .then(() => console.log('Recipe shared successfully'))
        .catch(error => console.error('Error sharing recipe:', error));
    } else {
        // Fallback for browsers that don't support Web Share API
        const tempInput = document.createElement('input');
        document.body.appendChild(tempInput);
        tempInput.value = window.location.href;
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        
        alert('Recipe URL copied to clipboard!');
    }
}

// Function to add voice search capability
function setupVoiceSearch() {
    if ('webkitSpeechRecognition' in window) {
        const searchBox = document.querySelector('.search-box');
        const voiceButton = document.createElement('button');
        voiceButton.innerHTML = '🎤';
        voiceButton.title = 'Search by voice';
        voiceButton.style.backgroundColor = '#ff4081';
        voiceButton.style.borderRadius = '50%';
        voiceButton.style.width = '40px';
        voiceButton.style.height = '40px';
        voiceButton.style.display = 'flex';
        voiceButton.style.alignItems = 'center';
        voiceButton.style.justifyContent = 'center';
        voiceButton.onclick = startVoiceSearch;
        searchBox.appendChild(voiceButton);
    }
}

// Voice search function
function startVoiceSearch() {
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = function() {
            document.getElementById('query').placeholder = 'Listening...';
        };
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('query').value = transcript;
            document.getElementById('query').placeholder = 'e.g., chicken, pasta, vegetarian dinner';
            setTimeout(generateRecipe, 500);
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error', event.error);
            document.getElementById('query').placeholder = 'e.g., chicken, pasta, vegetarian dinner';
        };
        
        recognition.onend = function() {
            document.getElementById('query').placeholder = 'e.g., chicken, pasta, vegetarian dinner';
        };
        
        recognition.start();
    }
}

// Add event listener for Enter key in search box
document.getElementById('query').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        generateRecipe();
    }
});

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    // Load saved recipes and history on startup
    loadSavedRecipes();
    loadRecipeHistory();
    
    // Setup voice search if available
    setupVoiceSearch();
    
    // Add theme toggle
    addThemeToggle();
    
    // Check for dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
    }
});

// Add dark mode toggle
function addThemeToggle() {
    const container = document.querySelector('.container');
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = '🌙';
    themeToggle.title = 'Toggle dark mode';
    themeToggle.style.position = 'absolute';
    themeToggle.style.top = '20px';
    themeToggle.style.right = '20px';
    themeToggle.style.backgroundColor = 'transparent';
    themeToggle.style.border = 'none';
    themeToggle.style.fontSize = '20px';
    themeToggle.style.cursor = 'pointer';
    themeToggle.style.zIndex = '100';
    
    themeToggle.onclick = function() {
        document.body.classList.toggle('dark-mode');
        themeToggle.innerHTML = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    };
    
    // Update button state based on current theme
    if (localStorage.getItem('darkMode') === 'true') {
        themeToggle.innerHTML = '☀️';
    }
    
    document.body.insertBefore(themeToggle, container);
    
    // Add dark mode styles
    const darkModeStyles = document.createElement('style');
    darkModeStyles.textContent = `
        .dark-mode {
            background-color: #121212;
            color: #e0e0e0;
        }
        .dark-mode .container {
            background-color: #1e1e1e;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .dark-mode h1, .dark-mode h2, .dark-mode h3 {
            color: #4CAF50;
        }
        .dark-mode input, .dark-mode select {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-color: #444;
        }
        .dark-mode .preferences {
            background-color: #2d2d2d;
            border-color: #444;
        }
        .dark-mode .preference-options label {
            background-color: #3d3d3d;
            border-color: #555;
        }
        .dark-mode .recipe, .dark-mode .recipe-grid-item {
            background-color: #2d2d2d;
            border-color: #444;
        }
        .dark-mode .nutrition-info {
            background-color: #3d3d3d;
            border-color: #555;
        }
        .dark-mode .nutrition-item {
            background-color: #2d2d2d;
        }
        .dark-mode .nutrition-value {
            color: #e0e0e0;
        }
        .dark-mode .similar-recipe {
            background-color: #2d2d2d;
            border-color: #444;
        }
        .dark-mode .tab-button {
            color: #aaa;
        }
        .dark-mode .tab-button.active {
            color: #4CAF50;
        }
        .dark-mode a {
            color: #81c784;
        }
    `;
    document.head.appendChild(darkModeStyles);
}