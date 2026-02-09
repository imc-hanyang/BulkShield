import { BaseView } from './BaseView.js';
import { state } from '../state.js';
import { router } from '../Router.js';
import { db, collection, addDoc, ADMIN_EMAIL, COLLECTION_LLM } from '../firebase-config.js';

export class LLMAnalysisView extends BaseView {
    constructor() {
        super();
        this.data = null; // Placeholder for LLM data
    }

    async render() {
        this.clear();
        this.setDescription('');

        // Load HTML from template
        const template = await this.loadTemplate('./templates/llm-analysis.html');

        const container = document.createElement('div');
        container.innerHTML = template;

        this.appRoot.appendChild(container);

        // Bind Buttons
        this.submitBtn = document.getElementById('btn-submit-llm');
        this.backBtn = document.getElementById('btn-back-llm');
        this.commentBox = document.getElementById('user-comment');

        this.setupEvents();
        this.initCharts();
        this.renderEventLogs();
    }

    renderEventLogs() {
        const rawData = `Related Event[2024-09-06 17:54:36] REFUND | 1 Seat | 40,700 KRW GwangjuSongjeong → Suseo (Dep: 2024-09-06 (Fri) 17:57) <Seat occupied for 3,920 mins after purchase> <Refunded 23 mins before departure> <Adjacent seat refunded>(Post-work return demand time)[2024-09-13 18:24:12] REFUND | 1 Seat | 40,700 KRW GwangjuSongjeong → Suseo (Dep: 2024-09-13 (Fri) 18:31) <Seat occupied for 4,110 mins after purchase> <Refunded 7 mins before departure> <Adjacent seat refunded>[2024-09-20 19:52:48] REFUND | 1 Seat | 40,700 KRW GwangjuSongjeong → Suseo (Dep: 2024-09-20 (Fri) 19:54) <Seat occupied for 4,260 mins after purchase> <Refunded 2 mins before departure> <Adjacent seat refunded>[2024-09-27 21:00:33] REFUND | 2 Seats | 81,400 KRW GwangjuSongjeong → Suseo (Dep: 2024-09-27 (Fri) 21:10) <Seat occupied for 4,180 mins after purchase> <Refunded 10 mins before departure> <Adjacent seat refunded>[2024-09-24 18:30:55] PURCHASE | 3 Seats | 122,100 KRW Suseo → GwangjuSongjeong (Dep: 2024-09-30 (Mon) 05:08) <Cumulative reverse direction purchases: 5> <Min gap between arrival and reverse ticket departure: 6,487 mins>`;

        const container = document.getElementById('event-logs-container');
        if (!container) return;

        // Remove "Related Event" prefix if exists
        let content = rawData.replace(/^Related Event/, '').trim();

        // Split by timestamp pattern, but keep the timestamp
        // Regex Lookahead to split before [YYYY-MM-DD...]
        const parts = content.split(/(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])/g);

        container.innerHTML = parts.map(part => {
            if (!part.trim()) return '';

            // Extract Timestamp [YYYY...]
            const timeMatch = part.match(/^\[(.*?)\]/);
            if (!timeMatch) return '';
            const timestamp = timeMatch[0];
            const timeStr = timeMatch[1];

            // Rest of the string after timestamp
            let remainder = part.substring(timestamp.length).trim();

            // Extract Tags <...>
            const tags = [];
            const tagRegex = /<([^>]+)>/g;
            let match;

            // We need to extract the "Route Info" which is everything before the first tag
            const firstTagIndex = remainder.indexOf('<');
            let routeInfo = '';

            if (firstTagIndex !== -1) {
                routeInfo = remainder.substring(0, firstTagIndex).trim();
                // Find all tags
                while ((match = tagRegex.exec(remainder)) !== null) {
                    tags.push(match[1]); // content inside <>
                }
            } else {
                routeInfo = remainder;
            }

            // Generate HTML
            const tagHtml = tags.map(tag => `<span class="warning">${tag}</span>`).join('');

            return `
                <div class="log-entry">
                    <span class="timestamp">${timestamp}</span>
                    <span class="route">${routeInfo}</span>
                    <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 8px;">
                        ${tagHtml}
                    </div>
                </div>
            `;
        }).join('');
    }

    async initCharts() {
        // Render Fixed Stats immediately
        this.renderStats();

        try {
            // Fetch CSV
            const response = await fetch('./data.csv');
            if (!response.ok) throw new Error('Failed to load data.csv');
            const csvText = await response.text();

            // Parse CSV
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    this.processAndRenderCharts(results.data);
                }
            });
        } catch (error) {
            console.warn("Chart Data Load Error (Charts may not appear):", error);
        }
    }

    renderStats() {
        // OVERRIDE STATS WITH USER PROVIDED VALUES (Hardcoded for Demo/Analysis)
        const totalPurchaseCount = 18;
        const totalPurchaseAmount = 857400;
        const totalRefundCount = 10;
        const totalRefundAmount = 468900;

        // Date Range (Mocked or omitted if no data)
        const dateRangeEl = document.getElementById('analysis-date-range');
        if (dateRangeEl) dateRangeEl.textContent = "Last 4 Weeks";

        const statsHeaderEl = document.getElementById('summary-stats-header');
        if (statsHeaderEl) {
            statsHeaderEl.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-box purchase">
                        <span class="stat-label">🎫 Total Purchase</span>
                        <span class="stat-value-main">${totalPurchaseAmount.toLocaleString()} KRW</span>
                        <span class="stat-value-sub">${totalPurchaseCount} Cases</span>
                    </div>
                    <div class="stat-box refund">
                        <span class="stat-label">↩️ Total Refund</span>
                        <span class="stat-value-main">${totalRefundAmount.toLocaleString()} KRW</span>
                        <span class="stat-value-sub">${totalRefundCount} Cases</span>
                    </div>
                </div>
            `;
        }
    }

    processAndRenderCharts(data) {
        // Data format: { timestamp, event_type, amount_krw, ... }
        // We need to aggregate by Date (YYYY-MM-DD)
        const dailyData = {};

        // Use the last row's timestamp as the reference date (Action Date)
        let lastRow = data[data.length - 1];
        // Handle case where last row might be empty (e.g. trailing newline)
        if (!lastRow || !lastRow.timestamp) {
            // Try searching backwards for valid data
            for (let i = data.length - 1; i >= 0; i--) {
                if (data[i].timestamp) {
                    lastRow = data[i];
                    break;
                }
            }
        }

        let maxDate = new Date(); // Default to today
        if (lastRow && lastRow.timestamp) {
            maxDate = new Date(lastRow.timestamp);
        }

        // Stats Calculation Logic removed from here as we use hardcoded values in renderStats

        let minTime = null;
        let maxTime = null;

        data.forEach(row => {
            if (!row.timestamp || !row.amount_krw) return;

            // Date Range
            const rowDate = new Date(row.timestamp);
            if (!isNaN(rowDate)) {
                if (!minTime || rowDate < minTime) minTime = rowDate;
                if (!maxTime || rowDate > maxTime) maxTime = rowDate;
            }

            const amount = parseInt(row.amount_krw, 10) || 0;

            // Chart Data Aggregation
            if (isNaN(rowDate)) return;
            const labelKey = `${rowDate.getMonth() + 1}/${rowDate.getDate()}`; // M/D format

            if (!dailyData[labelKey]) {
                dailyData[labelKey] = { purchase: 0, refund: 0 };
            }

            if (row.event_type === 'PURCHASE') {
                dailyData[labelKey].purchase += amount;
            } else if (row.event_type === 'REFUND') {
                dailyData[labelKey].refund += amount;
            }
        });

        // Update Date Range if real data exists
        if (minTime && maxTime) {
            const formatMonthDay = (d) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            const dateRangeText = `${formatMonthDay(minTime)} and ${formatMonthDay(maxTime)}, ${maxTime.getFullYear()}`;
            const dateRangeEl = document.getElementById('analysis-date-range');
            if (dateRangeEl) dateRangeEl.textContent = dateRangeText;
        }

        // Generate Labels and Data for the last 28 days ending at maxDate (or today if desired, but using maxDate for context)
        // If maxDate is invalid (no data), default to today
        if (maxDate.getTime() === 0) maxDate = new Date();

        const labels = [];
        const bookingAmounts = [];
        const refundAmounts = [];

        for (let i = 27; i >= 0; i--) {
            const date = new Date(maxDate);
            date.setDate(maxDate.getDate() - i);
            const labelKey = `${date.getMonth() + 1}/${date.getDate()}`;

            labels.push(labelKey);

            if (dailyData[labelKey]) {
                bookingAmounts.push(dailyData[labelKey].purchase);
                refundAmounts.push(dailyData[labelKey].refund);
            } else {
                bookingAmounts.push(0);
                refundAmounts.push(0);
            }
        }

        this.renderCharts(labels, bookingAmounts, refundAmounts);
    }

    renderCharts(labels, bookingAmounts, refundAmounts) {
        // Chart 1: Booking Amount (Bar Chart)
        const ctx1 = document.getElementById('chart-booking-amount').getContext('2d');
        if (this.chart1) this.chart1.destroy();

        this.chart1 = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Purchase Amount (Daily)',
                    data: bookingAmounts,
                    backgroundColor: 'rgba(59, 130, 246, 0.7)', // Blue
                    borderColor: '#3b82f6',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Daily Purchase Amount Trend` },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Amount (KRW)' }
                    }
                }
            }
        });

        // Chart 2: Refund Amount (Bar Chart)
        const ctx2 = document.getElementById('chart-refund-amount').getContext('2d');
        if (this.chart2) this.chart2.destroy();

        this.chart2 = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Refund Amount (Daily)',
                    data: refundAmounts,
                    backgroundColor: 'rgba(239, 68, 68, 0.7)', // Red
                    borderColor: '#ef4444',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Daily Refund Amount Trend` },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Amount (KRW)' }
                    }
                }
            }
        });
    }

    setupEvents() {
        this.backBtn.addEventListener('click', () => {
            router.navigate('/task-selection');
        });

        this.submitBtn.addEventListener('click', () => this.submitData());

        // Toggle Views
        // Note: Elements are looked up here because they are inside the innerHTML which was set in render.
        // However, this.setupEvents is called after template load.
        const chartsView = document.getElementById('charts-view');
        const detailsView = document.getElementById('details-view');
        const btnShowDetails = document.getElementById('btn-show-details');
        const btnShowSummary = document.getElementById('btn-show-summary'); // Close button inside details

        if (btnShowDetails && btnShowSummary) {
            btnShowDetails.addEventListener('click', () => {
                chartsView.classList.add('hidden');
                detailsView.classList.remove('hidden');
            });

            btnShowSummary.addEventListener('click', () => {
                detailsView.classList.add('hidden');
                chartsView.classList.remove('hidden');
            });
        }

        // Validation Listeners
        const radios = document.querySelectorAll('input[type="radio"]');
        radios.forEach(r => r.addEventListener('change', () => this.validateForm()));

        this.commentBox.addEventListener('input', () => this.validateForm());
    }

    validateForm() {
        // 1. Check if all 6 questions are answered
        const questions = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'];
        const allAnswered = questions.every(q => {
            return document.querySelector(`input[name="${q}"]:checked`);
        });

        // 2. Check if comment is not empty
        const hasComment = this.commentBox.value.trim().length > 0;

        // Enable/Disable Submit Button
        if (allAnswered && hasComment) {
            this.submitBtn.disabled = false;
        } else {
            this.submitBtn.disabled = true;
        }
    }

    async submitData() {
        if (this.submitBtn.disabled) return;
        if (!confirm('Do you want to submit the analysis results?')) return;

        try {
            // Collect Survey Data
            const surveyResults = {
                q1: document.querySelector('input[name="q1"]:checked').value,
                q2: document.querySelector('input[name="q2"]:checked').value,
                q3: document.querySelector('input[name="q3"]:checked').value,
                q4: document.querySelector('input[name="q4"]:checked').value,
                q5: document.querySelector('input[name="q5"]:checked').value,
                q6: document.querySelector('input[name="q6"]:checked').value,
                comment: this.commentBox.value.trim()
            };

            // Include User Info in Submission
            const resultData = {
                user_info: state.userInfo,
                labeler_email: ADMIN_EMAIL,
                timestamp: new Date(),
                task_type: 'llm_analysis',
                survey_results: surveyResults
            };

            await addDoc(collection(db, COLLECTION_LLM), resultData);

            alert("Submission Completed.");

            // Mark task as complete in global state
            state.completeTask('llmAnalysis');

            // Return to Task Selection
            router.navigate('/task-selection');

        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        }
    }
}
