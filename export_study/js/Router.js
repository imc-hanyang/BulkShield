export class Router {
    constructor() {
        this.routes = {};
        this.currentView = null;

        // Bind methods
        this.handleRoute = this.handleRoute.bind(this);

        // Listen for hash changes and initial load
        window.addEventListener('hashchange', this.handleRoute);
        window.addEventListener('load', this.handleRoute);
    }

    /**
     * Register a route
     * @param {string} path - The hash path (e.g., '/intro')
     * @param {class} ViewClass - The View class to render
     */
    addRoute(path, ViewClass) {
        this.routes[path] = ViewClass;
    }

    /**
     * Navigate to a path programmatically
     * @param {string} path 
     */
    navigate(path) {
        window.location.hash = path;
    }

    /**
     * Handle routing based on current hash
     */
    async handleRoute() {
        // Get path from hash (remove #)
        let path = window.location.hash.slice(1) || '/';

        // Default redirect to /intro if root
        if (path === '/' || path === '') {
            path = '/intro';
            // Only replace if we are literally at root, 
            // otherwise navigate() pushes history which is fine.
            // But for initial load sync:
            if (window.location.hash !== '#/intro') {
                window.location.hash = '/intro';
                return; // hashchange will trigger re-handle
            }
        }

        const ViewClass = this.routes[path];

        if (ViewClass) {
            // Cleanup current view if needed (optional, implemented in BaseView?)
            // render() typically clears appRoot first.

            console.log(`Navigating to: ${path}`);
            this.currentView = new ViewClass();
            await this.currentView.render();
        } else {
            console.warn(`No route found for: ${path}`);
            // Optional: 404 View or redirect
            this.navigate('/intro');
        }
    }
}

// Create and export singleton instance
export const router = new Router();
