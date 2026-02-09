import { BaseView } from './BaseView.js';

export class CompletionView extends BaseView {
    async render() {
        this.clear();
        this.setDescription(''); // No description needed for final screen

        const container = document.createElement('div');
        container.style.textAlign = 'center';
        container.style.marginTop = '100px';

        // Load HTML from template
        const template = await this.loadTemplate('./templates/completion.html');
        container.innerHTML = template;

        this.appRoot.appendChild(container);
    }
}
