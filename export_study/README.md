# Expert Data Labeling System for Mass Refund Risk Detection

A web-based expert survey application designed to validate AI-driven mass refund risk detection models for SRT (Korea's high-speed rail service). This system enables domain experts to evaluate LLM-generated interpretations and verify risk classification criteria through structured surveys.

## Overview

This application supports a research study on **preemptive detection of mass refund potential risks** in train ticket booking systems. The goal is to move beyond post-hoc, rule-based punishment methods by analyzing user booking patterns to identify users with a high probability of mass refunds, thereby preventing seat hoarding and ensuring fair seat access.

The system collects expert evaluations through two primary tasks:

1. **LLM Analysis Evaluation**: Experts assess whether AI-generated explanations of user booking/refund behaviors are practically usable for identifying mass refund risks.

2. **Illegal Booking Verification**: Experts classify individual users as requiring sanction, further interpretation, or no action based on their 4-week booking and refund statistics.

## Features

- **Single-Page Application (SPA)** architecture with hash-based routing
- **Firebase Integration** for secure data persistence
- **Interactive Visualizations** using Chart.js for purchase/refund trend analysis
- **Responsive Design** optimized for both desktop and mobile devices
- **Progress Tracking** with real-time validation and submission controls
- **Likert Scale Surveys** for structured expert feedback collection

## Technology Stack

- **Frontend**: Vanilla JavaScript (ES6 Modules), HTML5, CSS3
- **Backend/Database**: Firebase Firestore & Authentication
- **Visualization**: Chart.js 4.x
- **Data Processing**: PapaParse (CSV parsing)
- **Icons**: Feather Icons
- **Fonts**: Noto Sans KR (Google Fonts)

## Project Structure

```
export_study2/
├── index.html                 # Main entry point
├── style.css                  # Global styles and responsive design
├── data.csv                   # Sample booking/refund event data for LLM analysis
├── illegalBookingViewData.csv # User statistics for illegal booking verification
├── js/
│   ├── main.js                # Application bootstrap and route registration
│   ├── Router.js              # Hash-based SPA router implementation
│   ├── state.js               # Global state management
│   ├── firebase-config.js     # Firebase configuration and exports
│   └── views/
│       ├── BaseView.js        # Abstract base class for all views
│       ├── IntroView.js       # Research introduction and consent
│       ├── UserInfoView.js    # Participant information collection
│       ├── TaskSelectionView.js  # Task navigation hub
│       ├── IllegalBookingView.js # Booking verification survey
│       ├── LLMAnalysisView.js    # LLM interpretation evaluation
│       └── CompletionView.js     # Survey completion confirmation
└── templates/
    ├── intro.html             # Study introduction content
    ├── user-info.html         # Participant form template
    ├── task-selection.html    # Task selection cards
    ├── illegal-booking.html   # Booking verification interface
    ├── llm-analysis.html      # LLM analysis evaluation interface
    └── completion.html        # Thank you message
```

## Getting Started

### Prerequisites

- A modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.x (for local development server)
- Internet connection (required for Firebase and CDN resources)

### Installation

1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   cd export_study
   ```

2. Start a local development server:
   ```bash
   python -m http.server 3030
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3030
   ```

### Firebase Configuration

The application uses Firebase for authentication and data storage. To use your own Firebase project:

1. Create a new Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Email/Password authentication
3. Create a Firestore database
4. Update the configuration in `js/firebase-config.js`:
   ```javascript
   const firebaseConfig = {
       apiKey: "YOUR_API_KEY",
       authDomain: "YOUR_PROJECT.firebaseapp.com",
       projectId: "YOUR_PROJECT_ID",
       storageBucket: "YOUR_PROJECT.firebasestorage.app",
       messagingSenderId: "YOUR_SENDER_ID",
       appId: "YOUR_APP_ID"
   };
   ```
5. Add your domain to Firebase Console > Authentication > Settings > Authorized Domains

## Application Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│    Intro    │────▶│  User Info  │────▶│ Task Selection  │
│   (Study    │     │  (Name,     │     │                 │
│  Overview)  │     │  Demographics)│   │                 │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         │                       │                       │
                         ▼                       │                       ▼
               ┌─────────────────┐               │             ┌─────────────────┐
               │  LLM Analysis   │               │             │ Illegal Booking │
               │  Evaluation     │◀──────────────┘             │  Verification   │
               │  (Must complete │                             │  (Unlocks after │
               │   first)        │                             │   LLM task)     │
               └────────┬────────┘                             └────────┬────────┘
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
                                      ┌─────────────────┐
                                      │   Completion    │
                                      │  (Thank You)    │
                                      └─────────────────┘
```

## Data Format

### LLM Analysis Data (`data.csv`)

Contains individual booking and refund events for a sample user:

| Column | Description |
|--------|-------------|
| `timestamp` | Event timestamp (YYYY-MM-DD HH:MM:SS) |
| `event_type` | PURCHASE or REFUND |
| `route` | Origin → Destination |
| `seat_cnt` | Number of seats |
| `amount_krw` | Transaction amount in KRW |
| `departure_time` | Scheduled departure time |
| `hold_time_min` | Minutes seat was held before refund |
| `refund_before_dep_min` | Minutes before departure when refunded |

### Illegal Booking Data (`illegalBookingViewData.csv`)

Contains aggregated 4-week statistics for each user:

| Column | Description |
|--------|-------------|
| `id` | Sequential identifier |
| `user_id` | Anonymized user hash |
| `group` | User group classification |
| `total_ticket_count` | Total tickets purchased |
| `total_refund_count` | Total refunds made |
| `total_refund_amount` | Total refund amount (KRW) |
| `refund_rate` | Refund rate (0.0 - 1.0) |
| `cluster` | Statistical cluster assignment |

## Survey Tasks

### Task 1: LLM Analysis Evaluation

Experts review an AI-generated interpretation of a detected at-risk user, including:
- Daily purchase/refund amount charts
- Detailed event logs with flagged risk indicators
- Risk justification and booking intent interpretation

The survey includes 6 Likert-scale questions assessing:
1. Clarity of explanation
2. Relevance of identified patterns
3. Overall confidence in the analysis
4. Practical utility for decision-making
5. Potential for confusion or misunderstanding
6. Intent to adopt similar tools in practice

### Task 2: Illegal Booking Verification

Experts review 30 users with their 4-week booking statistics and classify each as:
- **○ (Circle)**: Sanction Required
- **△ (Triangle)**: Booking/Refund Intention Interpretation Required
- **✕ (X)**: Interpretation Not Required (Normal User)

## Firebase Collections

Survey results are stored in two Firestore collections:

- `llm_analysis_results`: LLM evaluation responses with survey answers
- `booking_verification_results`: Illegal booking classifications with user decisions

Each document includes:
- User demographic information
- Timestamp
- Survey responses or classification results
- Labeler email

## Browser Compatibility

The application uses ES6 modules and modern JavaScript features. Supported browsers:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

## License

This project is part of an academic research study. Please contact the research team for licensing information.

## Acknowledgments

- SRT (Korea Train Express) for domain expertise and collaboration

