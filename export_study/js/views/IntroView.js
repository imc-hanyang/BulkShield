import { BaseView } from './BaseView.js';
import { router } from '../Router.js';

export class IntroView extends BaseView {
    async render() {
        this.clear();
        this.setDescription('');

        const container = document.createElement('div');
        container.className = 'intro-container fade-in';

        // Load HTML from template
        const template = await this.loadTemplate('./templates/intro.html');
        container.innerHTML = template;

        this.appRoot.appendChild(container);

        // Event Listener
        container.querySelector('#btn-start').addEventListener('click', () => {
            router.navigate('/user-info');
        });
    }
}
