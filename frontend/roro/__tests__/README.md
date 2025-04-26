
# Roro Chatbot - Frontend Component Testing

## 📚 Overview
This README provides instructions for setting up, running, and understanding the frontend component tests for the Roro Chatbot project.  
The tests focus on core components of the chatbot such as the Chat Page, Dropdown, and Home Page.

## 💻 System Requirements
- **Operating System**: macOS (M2 Chip)
- **Node.js**: Version 18.x or higher
- **npm**: Version 9.x or higher
- **Testing Libraries**: React Testing Library, Jest

## 📂 File Structure
The testing code and related files are located in the following directory structure:

```
capstone-project-2025-t1-25t1-9900-t18a-brioche/
└── frontend/
    └── roro/
        ├── __mocks__/
        ├── __tests__/
        │   ├── chatPage.test.tsx
        │   ├── chatPageError.test.tsx
        │   ├── chatPolling.test.tsx
        │   ├── dropdown.test.tsx
        │   ├── home.test.tsx
        │   ├── homeError.test.tsx
        │   ├── sanity.test.tsx
        │   └── roro_component_test_report.md
```

- **chatPage.test.tsx**: Tests the functionality of the Chat Page (message sending and receiving).
- **chatPageError.test.tsx**: Tests error handling on the Chat Page.
- **chatPolling.test.tsx**: Verifies polling behavior.
- **dropdown.test.tsx**: Tests hover and click interactions of the Dropdown component.
- **home.test.tsx**: Verifies integration of components on the Home Page.
- **homeError.test.tsx**: Tests error handling on the Home Page.
- **sanity.test.tsx**: Basic test to confirm the environment is properly set up.

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone <url>
```

### Step 2: Install Dependencies
Navigate to the roro directory:
```bash
cd frontend/roro
```

Install the required dependencies:
```bash
npm install
```

### Step 3: Install Testing Libraries
```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom jest
npm install --save-dev ts-jest babel-jest @babel/preset-env @babel/preset-typescript
```

## 🧪 Running the Tests

To run all tests:
```bash
npx jest
```

### Running a Specific Test File
```bash
npx jest __tests__/chatPage.test.tsx
```

---

## ✅ Test Result Example
When the tests run successfully, you should see output like:

```
PASS  __tests__/dropdown.test.tsx
PASS  __tests__/chatPage.test.tsx
PASS  __tests__/home.test.tsx
...
Test Suites: 7 passed, 7 total
Tests: 12 passed, 12 total
Time: 1.48 s
```

---

## 📊 Test Breakdown

| Component         | What Was Tested                                           | Why It Matters |
|:------------------|:-----------------------------------------------------------|:---------------|
| **Chat Page**      | User input, assistant reply, and polling behavior          | Ensures smooth conversation flow and polling stops correctly. |
| **Dropdown**       | Hover and click interactions, dropdown rendering, event triggers | Ensures dropdown displays and triggers correct prompts. |
| **Home Page**      | End-to-end user flow from dropdown to chat initiation      | Validates that the user can initiate chat correctly from Home Page. |
| **Error Handling** | Handling failed API requests                              | Ensures app does not crash and handles API errors gracefully. |

---

## 🛠 Mocking External Libraries
- **Axios**: Mocked to simulate API requests and responses.
- **React-Markdown**: Mocked to avoid import/rendering issues.
- **Next.js useRouter and useParams**: Mocked to simulate navigation and route parameters.

---

## 🎯 Conclusion
The tests ensure that the Roro Chatbot's frontend components behave as expected, focusing on smooth interactions for users, particularly international students.  
By mocking external dependencies like API calls and route navigation, we isolate components and validate functionality without relying on live services.
