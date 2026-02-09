import { BaseView } from './BaseView.js';
import { state } from '../state.js';
import { router } from '../Router.js';

export class UserInfoView extends BaseView {
    constructor() {
        super();
    }

    async render() {
        this.clear();
        this.setDescription('Please enter participant information.');

        const formContainer = document.createElement('div');
        formContainer.className = 'form-container';

        // Load HTML from template
        const template = await this.loadTemplate('./templates/user-info.html');
        formContainer.innerHTML = template;

        this.appRoot.appendChild(formContainer);

        document.getElementById('btn-male').addEventListener('click', () => this.handleGenderSelect('male'));
        document.getElementById('btn-female').addEventListener('click', () => this.handleGenderSelect('female'));

        document.getElementById('btn-start').addEventListener('click', () => this.handleNext());
    }

    handleGenderSelect(gender) {
        this.selectedGender = gender;
        const btnMale = document.getElementById('btn-male');
        const btnFemale = document.getElementById('btn-female');

        // Reset classes
        btnMale.classList.remove('selected-male');
        btnFemale.classList.remove('selected-female');

        // Apply classes
        if (gender === 'male') {
            btnMale.classList.add('selected-male');
        } else if (gender === 'female') {
            btnFemale.classList.add('selected-female');
        }
    }

    handleNext() {
        // ... (rest is same, but reusing existing method logic)
        const name = document.getElementById('input-name').value.trim();
        const degree = document.getElementById('input-degree').value.trim();
        const age = document.getElementById('input-age').value.trim();
        const job = document.getElementById('input-job').value.trim();
        const gender = this.selectedGender;

        if (!name || !degree || !age || !job || !gender) {
            alert('Please enter all information (including name and gender).');
            return;
        }

        // Save to state
        // Use a fallback for UUID generation since crypto.randomUUID requires HTTPS
        const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        state.setUserInfo({ name, degree, age, job, gender, uuid });

        // Navigate to Selection View
        router.navigate('/task-selection');
    }
}
