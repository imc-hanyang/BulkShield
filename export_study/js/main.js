import { router } from './Router.js';
import { IntroView } from './views/IntroView.js';
import { UserInfoView } from './views/UserInfoView.js';
import { TaskSelectionView } from './views/TaskSelectionView.js';
import { IllegalBookingView } from './views/IllegalBookingView.js';
import { LLMAnalysisView } from './views/LLMAnalysisView.js';
import { auth, ADMIN_EMAIL, ADMIN_PWD } from './firebase-config.js';
import { signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Attempt auto-login
    signInWithEmailAndPassword(auth, ADMIN_EMAIL, ADMIN_PWD)
        .then((userCredential) => {
            console.log("✅ Login Success:", userCredential.user.email);

            // Register Routes
            router.addRoute('/intro', IntroView);
            router.addRoute('/user-info', UserInfoView);
            router.addRoute('/task-selection', TaskSelectionView);
            router.addRoute('/illegal-booking', IllegalBookingView);
            router.addRoute('/llm-analysis', LLMAnalysisView);

            // Initialize Routing (checks current hash or defaults to /intro)
            router.handleRoute();
        })
        .catch((error) => {
            console.error("Login Error:", error);
            const appRoot = document.getElementById('app-root');
            if (appRoot) {
                appRoot.innerHTML = `
                    <div style="text-align:center; padding: 40px; color: #ef4444;">
                        <h2>⚠️ Initialization Error</h2>
                        <p>Login Failed.</p>
                        <p style="background:#eee; padding:10px; border-radius:4px; margin:10px 0;">${error.message}</p>
                        <p style="font-size:0.9rem; color:#666;">
                            IP address (${location.hostname}) may not be in the Firebase authorized domains.<br>
                            Please add the current IP to Firebase Console > Authentication > Settings > Authorized Domains.
                        </p>
                        <button class="btn btn-secondary mt-4" onclick="location.reload()">Refresh</button>
                    </div>
                `;
            }
        });
});
