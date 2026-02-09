// Firebase Setup
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
export { collection, addDoc };
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyByyzGjMK1okYuSWY4Jvh-prZkavLq0Pkk",
    authDomain: "srt-export-test.firebaseapp.com",
    projectId: "srt-export-test",
    storageBucket: "srt-export-test.firebasestorage.app",
    messagingSenderId: "183020505952",
    appId: "1:183020505952:web:0fbb38bd7cee52b35bec05"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const auth = getAuth(app);

// Credentials
export const ADMIN_EMAIL = "imc@google.com";
export const ADMIN_PWD = "1234qwer";

// Config
// 1. Collection for Illegal Booking Verification
export const COLLECTION_BOOKING = "booking_verification_results";
// 2. Collection for LLM Analysis Verification
export const COLLECTION_LLM = "llm_analysis_results";

