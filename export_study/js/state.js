// Global State
export const state = {
    userInfo: {
        name: '',
        degree: '',
        age: '',
        gender: '',
        job: ''
    },

    // Status of tasks
    tasks: {
        illegalBooking: {
            completed: false,
            data: null // Optional: store local results before final submit if needed (though we submit per task)
        },
        llmAnalysis: {
            completed: false,
            data: null
        }
    },

    setUserInfo(info) {
        this.userInfo = { ...this.userInfo, ...info };
    },

    completeTask(taskName) {
        if (this.tasks[taskName]) {
            this.tasks[taskName].completed = true;
        }
    },

    isAllTasksCompleted() {
        return this.tasks.illegalBooking.completed && this.tasks.llmAnalysis.completed;
    }
};
