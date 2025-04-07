module.exports = {
    transform: {
      '^.+\\.(ts|tsx)$': 'babel-jest',
    },
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
    moduleNameMapper: {
      '^@/(.*)$': '<rootDir>/src/$1',
    },
    transformIgnorePatterns: [
      '/node_modules/(?!react-markdown|remark-gfm)/',
    ],
    testMatch: ['**/__tests__/**/*.test.ts?(x)'],
  };
  
  
  