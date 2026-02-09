import { BaseView } from './BaseView.js';
import { state } from '../state.js';
import { router } from '../Router.js';
import { db, collection, addDoc, ADMIN_EMAIL, COLLECTION_BOOKING } from '../firebase-config.js';

export class IllegalBookingView extends BaseView {
    constructor() {
        super();
        this.localState = {
            allData: [],
            answers: {},
            page: 1,
            limit: 10,
            totalPages: 0
        };
    }

    async render() {
        this.clear();
        this.setDescription('');

        // Load HTML from template
        const template = await this.loadTemplate('./templates/illegal-booking.html');

        const container = document.createElement('div');
        container.innerHTML = template;
        this.appRoot.appendChild(container);

        // Bind Elements
        this.els = {
            loading: container.querySelector('#loading-spinner'),
            workSection: container.querySelector('#work-section'),
            dataContainer: container.querySelector('#data-container'),
            prevBtn: container.querySelector('#prev-btn'),
            nextBtn: container.querySelector('#next-btn'),
            submitBtn: container.querySelector('#submit-btn'),
            currPage: container.querySelector('#current-page'),
            totalPage: container.querySelector('#total-pages'),
            progressBar: container.querySelector('#progress-bar'),
            progressText: container.querySelector('#progress-text')
        };

        this.loadCSVData();
        this.setupEvents();
    }

    async loadCSVData() {
        try {
            const CACHE_KEY = 'csv_data_cache';
            const CACHE_duration = 5 * 60 * 1000; // 5분 (밀리초 단위)

            // 1. 캐시 확인
            const cached = localStorage.getItem(CACHE_KEY);
            let csvText = null;

            if (cached) {
                const { timestamp, data } = JSON.parse(cached);
                const now = new Date().getTime();

                // 5분이 안 지났으면 캐시 사용
                if (now - timestamp < CACHE_duration) {
                    console.log(`Using cached data (Age: ${Math.round((now - timestamp) / 1000)}s)`);
                    csvText = data;
                }
            }

            // 2. 캐시가 없거나 만료되었으면 새로 요청
            if (!csvText) {
                console.log("Fetching new data...");
                const timestamp = new Date().getTime();
                const response = await fetch(`./illegalBookingViewData.csv?t=${timestamp}`);

                if (!response.ok) throw new Error("Cannot find CSV file.");
                csvText = await response.text();

                // 캐시 저장
                localStorage.setItem(CACHE_KEY, JSON.stringify({
                    timestamp: new Date().getTime(),
                    data: csvText
                }));
            }

            // 3. 파싱 및 로드
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (!results.data || results.data.length === 0) {
                        alert("No data available.");
                        return;
                    }

                    this.localState.allData = results.data.map((row, idx) => ({
                        ...row,
                        id: row.id || idx + 1
                    }));

                    this.localState.totalPages = Math.ceil(this.localState.allData.length / this.localState.limit);
                    this.els.totalPage.textContent = this.localState.totalPages;

                    this.els.loading.classList.add('hidden');
                    this.els.workSection.classList.remove('hidden');

                    this.renderPage(1);
                }
            });
        } catch (error) {
            alert("Loading Error: " + error.message);
        }
    }

    renderPage(pageNum) {
        this.localState.page = pageNum;
        this.els.dataContainer.innerHTML = '';

        const start = (pageNum - 1) * this.localState.limit;
        const end = pageNum * this.localState.limit;
        const pageData = this.localState.allData.slice(start, end);

        pageData.forEach(row => {
            const savedVal = this.localState.answers[row.id];

            const div = document.createElement('div');
            div.className = 'data-row';
            div.innerHTML = `
                <div class="center" data-tooltip="ID">${row.id}</div>
                <div class="center" data-tooltip="Purchase Count">${this.formatNumber(row.total_ticket_count)}</div>
                <div class="center" data-tooltip="Refund Count">${this.formatNumber(row.total_refund_count)}</div>
                <div class="center" data-tooltip="Refund Amount (1 Month)">${this.formatMoney(row['한 달 내 반환금액'] || row.total_refund_amount)}</div>
                <div class="center" data-tooltip="Refund Rate (1 Month)">${this.formatRate((row['한 달 내 환불율'] || row.refund_rate) * 100)}%</div>
                <div class="center">
                    <div class="radio-group">
                        <label class="radio-label opt-o">
                            <input type="radio" name="row-${row.id}" value="O" ${savedVal === 'O' ? 'checked' : ''}>
                            <i data-feather="circle"></i>
                        </label>
                        <label class="radio-label opt-triangle">
                            <input type="radio" name="row-${row.id}" value="△" ${savedVal === '△' ? 'checked' : ''}>
                            <i data-feather="triangle"></i>
                        </label>
                        <label class="radio-label opt-x">
                            <input type="radio" name="row-${row.id}" value="X" ${savedVal === 'X' ? 'checked' : ''}>
                            <i data-feather="x"></i>
                        </label>
                    </div>
                </div>
            `;

            div.querySelectorAll('input').forEach(input => {
                input.addEventListener('change', (e) => {
                    this.localState.answers[row.id] = e.target.value;
                    this.updateProgress();
                    this.checkPageCompletion(pageData);
                });
            });

            this.els.dataContainer.appendChild(div);
        });

        // Initialize Feather Icons
        if (window.feather) {
            feather.replace();
        }

        this.updateUI();
        this.checkPageCompletion(pageData);
        window.scrollTo(0, 0);
    }

    checkPageCompletion(pageData) {
        const isPageComplete = pageData.every(row => this.localState.answers[row.id]);
        this.els.nextBtn.disabled = !isPageComplete;

        if (this.localState.page === this.localState.totalPages) {
            const totalCount = this.localState.allData.length;
            const answerCount = Object.keys(this.localState.answers).length;
            this.els.submitBtn.disabled = (answerCount < totalCount);
        }
    }

    formatMoney(val) {
        return Number(val || 0).toLocaleString() + ' KRW';
    }

    formatNumber(val) {
        return Number(val || 0).toLocaleString();
    }

    formatRate(val) {
        const num = Number(val);
        if (isNaN(num)) return '-';
        // 반올림해서 소수점 2자리까지
        return num.toFixed(2);
    }

    updateUI() {
        this.els.currPage.textContent = this.localState.page;
        this.els.prevBtn.disabled = this.localState.page === 1;

        if (this.localState.page === this.localState.totalPages) {
            this.els.nextBtn.classList.add('hidden');
            this.els.submitBtn.classList.remove('hidden');
        } else {
            this.els.nextBtn.classList.remove('hidden');
            this.els.submitBtn.classList.add('hidden');
        }
    }

    updateProgress() {
        const total = this.localState.allData.length;
        const answered = Object.keys(this.localState.answers).length;
        const percent = Math.round((answered / total) * 100);
        this.els.progressBar.style.width = `${percent}%`;
        this.els.progressText.textContent = `${percent}%`;
    }

    setupEvents() {
        this.els.prevBtn.addEventListener('click', () => {
            if (this.localState.page > 1) this.renderPage(this.localState.page - 1);
        });

        this.els.nextBtn.addEventListener('click', () => {
            if (!this.els.nextBtn.disabled && this.localState.page < this.localState.totalPages) {
                this.renderPage(this.localState.page + 1);
            }
        });

        this.els.submitBtn.addEventListener('click', () => this.submitData());
    }

    async submitData() {
        if (this.els.submitBtn.disabled) return;

        const count = Object.keys(this.localState.answers).length;
        if (!confirm(`Do you want to submit ${count} cases?`)) return;

        try {
            // Include User Info in Submission
            const resultData = {
                user_info: state.userInfo, // { degree, age, job }
                labeler_email: ADMIN_EMAIL,
                timestamp: new Date(),
                total_count: count,
                results: this.localState.answers,
                task_type: 'illegal_booking_check'
            };

            await addDoc(collection(db, COLLECTION_BOOKING), resultData);

            alert("Submission Completed.");

            // Mark task as complete in global state
            state.completeTask('illegalBooking');

            // Return to Task Selection
            router.navigate('/task-selection');

        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        }
    }
}
