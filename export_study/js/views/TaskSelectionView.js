import { BaseView } from './BaseView.js';
import { state } from '../state.js';
import { router } from '../Router.js';
import { CompletionView } from './CompletionView.js';

export class TaskSelectionView extends BaseView {
    async render() {
        this.clear();
        this.setDescription('Please proceed with the survey on the left first.');

        const container = document.createElement('div');
        container.className = 'selection-container';

        // Load HTML from template
        const template = await this.loadTemplate('./templates/task-selection.html');
        container.innerHTML = template;

        this.appRoot.appendChild(container);

        // Query Elements
        const cardBooking = container.querySelector('#card-booking');
        const descBooking = container.querySelector('#desc-booking');
        const cardLLM = container.querySelector('#card-llm');
        const descLLM = container.querySelector('#desc-llm');
        const finishContainer = container.querySelector('#finish-container');
        const btnFinish = container.querySelector('#btn-finish');

        // Logic 1: LLM Review (Left - First)
        if (state.tasks.llmAnalysis.completed) {
            cardLLM.classList.add('completed', 'disabled');
            descLLM.textContent = 'Completed';
        } else {
            cardLLM.addEventListener('click', () => {
                router.navigate('/llm-analysis');
            });
        }

        // Logic 2: Illegal Booking (Right - Second)
        if (state.tasks.illegalBooking.completed) {
            cardBooking.classList.add('completed', 'disabled');
            descBooking.textContent = 'Completed';
        } else if (!state.tasks.llmAnalysis.completed) {
            // Disable if LLM not finished
            cardBooking.classList.add('disabled');
            // Optional: visual cue that it's locked
            cardBooking.style.opacity = '0.5';
            cardBooking.style.cursor = 'not-allowed';
            descBooking.innerHTML = 'Please complete the <br>left survey first.';
        } else {
            cardBooking.addEventListener('click', () => {
                router.navigate('/illegal-booking');
            });
        }

        // Logic 3: Finish Button
        if (state.isAllTasksCompleted()) {
            finishContainer.classList.remove('hidden');
            btnFinish.addEventListener('click', () => {
                new CompletionView().render();
            });
        }
    }
}
