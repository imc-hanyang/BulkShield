export class BaseView {
    constructor() {
        this.appRoot = document.getElementById('app-root');
        this.headerDesc = document.getElementById('header-description');
    }

    /**
     * Set the description text in the header
     * @param {string} text 
     */
    setDescription(text) {
        if (this.headerDesc) {
            this.headerDesc.textContent = text;
        }
    }

    /**
     * Clear the main container
     */
    clear() {
        this.appRoot.innerHTML = '';
    }

    /**
     * Load an HTML template from a URL
     * @param {string} url - Path to the HTML file
     * @returns {Promise<string>}
     */
    async loadTemplate(url) {
        try {
            // Cache busting for templates
            const version = new Date().getTime();
            const response = await fetch(`${url}?v=${version}`);
            if (!response.ok) throw new Error(`Failed to load template: ${url}`);
            return await response.text();
        } catch (error) {
            console.error(error);
            return '<div class="error">Error loading template</div>';
        }
    }

    /**
     * Render the view (to be implemented by subclasses)
     */
    render() {
        throw new Error('Render method must be implemented');
    }
}
