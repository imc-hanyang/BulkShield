// Firebase Setup
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
export { collection, addDoc };
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "apiKey",
    authDomain: "authDomain",
    projectId: "projectId",
    storageBucket: "storageBucket",
    messagingSenderId: "messagingSenderId",
    appId: "appId"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const auth = getAuth(app);

// Credentials
export const ADMIN_EMAIL = "ADMIN_EMAIL";
export const ADMIN_PWD = "ADMIN_PWD";

// Config
// 1. Collection for Illegal Booking Verification
export const COLLECTION_BOOKING = "booking_verification_results";
// 2. Collection for LLM Analysis Verification
export const COLLECTION_LLM = "llm_analysis_results";

