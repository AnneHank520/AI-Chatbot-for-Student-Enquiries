'use client'

import React, { useState, useEffect } from "react";
import axios from "axios";
import { useMutation, useQueryClient, useQuery } from "@tanstack/react-query";

// API response type definitions
interface StatusResponse {
  vector_db_loaded: boolean;
  sentences_count: number;
  keywords_index_count: number;
  url_index_count: number;
}

interface FileListResponse {
  files: string[];
}

interface UrlPreview {
  url: string;
  title: string;
  content: string;
  status: string;
}

export default function AdminDashboard() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'system' | 'pdf' | 'url'>('system');
  
  // PDF management states
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [fileToDelete, setFileToDelete] = useState<string>('');
  
  // URL management states
  const [url, setUrl] = useState('');
  const [urlPreview, setUrlPreview] = useState<UrlPreview | null>(null);
  
  // System status query
  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['apiStatus'],
    queryFn: async () => {
      const response = await axios.get('/api/status');
      return response.data;
    },
    refetchInterval: 300000, // refresh every 5 minutes
  });
  
  // PDF file list
  const { data: fileListData, isLoading: isLoadingFiles, error: fileError, refetch: refetchFiles } = useQuery({
    queryKey: ['pdfFiles'],
    queryFn: async () => {
      const response = await axios.get('/api/list-files');
      return response.data;
    }
  });
  
  // Upload PDF mutation
  const { mutate: uploadPDF, isPending: isUploading } = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return axios.post('/api/process-pdf', formData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pdfFiles'] });
      queryClient.invalidateQueries({ queryKey: ['apiStatus'] });
      setUploadFile(null);
      refetchFiles();
      refetchStatus();
    },
  });

  // Reprocess PDF mutation
  const { mutate: reprocessPDF, isPending: isReprocessing } = useMutation({
    mutationFn: async (filename: string) => {
      return axios.post('/api/reprocess-pdf', { filename });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['apiStatus'] });
      refetchStatus();
      refetchFiles();
    },
  });

  // Delete PDF content mutation
  const { mutate: deletePDFContent, isPending: isDeleting } = useMutation({
    mutationFn: async (filename: string) => {
      return axios.post('/api/delete-pdf-content', { filename });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['apiStatus'] });
      setFileToDelete('');
      refetchStatus();
      refetchFiles();
    },
  });
  
  // Extract URL content mutation
  const { mutate: extractUrl, isPending: isExtracting } = useMutation({
    mutationFn: async (url: string) => {
      const response = await axios.post('/api/extract-url-content', { url });
      return response.data;
    },
    onSuccess: (data: any) => {
      setUrlPreview(data);
    },
  });
  
  // Add URL to index mutation
  const { mutate: addUrlToIndex, isPending: isAddingUrl } = useMutation({
    mutationFn: async (url: string) => {
      return axios.post('/api/add-url-to-index', { url });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['apiStatus'] });
      setUrl('');
      setUrlPreview(null);
      refetchStatus();
    },
  });
  
  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadFile(e.target.files[0]);
    }
  };

  return (
    <div className="container mx-auto p-6">
      {/* Main content area */}
      <div className="bg-white rounded-xl shadow-md mb-6">
        <div className="flex border-b">
          <button 
            onClick={() => setActiveTab('system')}
            className={`px-6 py-3 text-sm font-medium ${
              activeTab === 'system' 
                ? 'border-b-2 border-[#7FCD89] text-[#183728]' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            System Status
          </button>
          <button 
            onClick={() => setActiveTab('pdf')}
            className={`px-6 py-3 text-sm font-medium ${
              activeTab === 'pdf' 
                ? 'border-b-2 border-[#7FCD89] text-[#183728]' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            PDF Management
          </button>
          <button 
            onClick={() => setActiveTab('url')}
            className={`px-6 py-3 text-sm font-medium ${
              activeTab === 'url' 
                ? 'border-b-2 border-[#7FCD89] text-[#183728]' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            URL Management
          </button>
        </div>
      </div>
      
      {/* System status page */}
      {activeTab === 'system' && (
        <div>
          <div className="mb-6 bg-white rounded-xl shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-[#183728]">System Status</h2>
              <button 
                onClick={() => refetchStatus()}
                className="px-4 py-2 bg-[#7FCD89] text-[#183728] rounded-md hover:bg-green-400 transition"
              >
                Refresh Status
              </button>
            </div>
            
            {statusData ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className={`p-4 rounded-lg ${statusData.vector_db_loaded ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                  <h3 className="text-sm font-medium">Vector Database</h3>
                  <p className="text-2xl font-bold mt-1">{statusData.vector_db_loaded ? "Online" : "Offline"}</p>
                </div>
                <div className="p-4 rounded-lg bg-blue-100 text-blue-800">
                  <h3 className="text-sm font-medium">Total Sentences</h3>
                  <p className="text-2xl font-bold mt-1">{statusData.sentences_count || 0}</p>
                </div>
                <div className="p-4 rounded-lg bg-indigo-100 text-indigo-800">
                  <h3 className="text-sm font-medium">URL Links</h3>
                  <p className="text-2xl font-bold mt-1">{statusData.url_index_count || 0}</p>
                </div>
                <div className="p-4 rounded-lg bg-purple-100 text-purple-800">
                  <h3 className="text-sm font-medium">Keywords</h3>
                  <p className="text-2xl font-bold mt-1">{statusData.keywords_index_count || 0}</p>
                </div>
              </div>
            ) : (
              <div className="flex justify-center items-center h-40">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#183728]"></div>
              </div>
            )}
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-semibold text-[#183728] mb-4">API Endpoint Status</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Endpoint</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {[
                    { endpoint: '/status', method: 'GET', status: 'Online', description: 'Check API status' },
                    { endpoint: '/process-pdf', method: 'POST', status: 'Online', description: 'Upload and process PDF files' },
                    { endpoint: '/reprocess-pdf', method: 'POST', status: 'Online', description: 'Reprocess existing PDF files' },
                    { endpoint: '/delete-pdf-content', method: 'POST', status: 'Online', description: 'Delete PDF content from database' },
                    { endpoint: '/list-files', method: 'GET', status: 'Online', description: 'List available PDF files' },
                    { endpoint: '/query', method: 'POST', status: 'Online', description: 'Search PDF content' },
                    { endpoint: '/extract-url-content', method: 'POST', status: 'Online', description: 'Extract URL content' },
                    { endpoint: '/add-url-to-index', method: 'POST', status: 'Online', description: 'Add URL content to database' }
                  ].map((api, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{api.endpoint}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          api.method === 'GET' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
                        }`}>
                          {api.method}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                          {api.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{api.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
                  </div>
                )}
      
      {/* PDF management page */}
      {activeTab === 'pdf' && (
        <div>
          <div className="mb-6 bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-semibold text-[#183728] mb-4">Upload New PDF</h2>
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-6">
              <div className="flex flex-col items-center">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-1 text-sm text-gray-600">
                  <label htmlFor="file-upload" className="font-medium text-[#183728] hover:text-green-700 cursor-pointer">
                    Upload file
                  </label> or drag and drop
                </p>
                <p className="text-xs text-gray-500">PDF files up to 10MB</p>
                <input id="file-upload" type="file" accept=".pdf" onChange={handleFileChange} className="sr-only" />
              </div>
              
              {uploadFile && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <svg className="h-6 w-6 text-[#183728]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">{uploadFile.name}</p>
                        <p className="text-sm text-gray-500">{Math.round(uploadFile.size / 1024)} KB</p>
                      </div>
                    </div>
                    <button
                      onClick={() => uploadFile && uploadPDF(uploadFile)}
                      disabled={isUploading}
                      className="px-4 py-2 bg-[#7FCD89] text-[#183728] rounded-md hover:bg-green-400 focus:outline-none disabled:opacity-50"
                    >
                      {isUploading ? 'Processing...' : 'Process PDF'}
                    </button>
                  </div>
                  </div>
                )}
              </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-[#183728]">PDF Library</h2>
              <button
                onClick={() => refetchFiles()}
                className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition flex items-center"
              >
                <svg className="w-4 h-4 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </button>
            </div>
            
            {isLoadingFiles ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#183728]"></div>
              </div>
            ) : fileError ? (
              <div className="p-4 bg-red-50 text-red-700 rounded-lg">
                Error loading files
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Filename</th>
                      <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {fileListData?.files && fileListData.files.length > 0 ? (
                      fileListData.files.map((file: string, index: number) => (
                        <tr 
                          key={index}
                          className={`${selectedFile === file ? 'bg-green-50' : 'hover:bg-gray-50'}`}
                          onClick={() => setSelectedFile(file)}
                        >
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <svg className="flex-shrink-0 h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                              </svg>
                              <div className="ml-4">
                                <div className="text-sm font-medium text-gray-900">{file}</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFile(file);
                                reprocessPDF(file);
                              }}
                              disabled={isReprocessing}
                              className="text-[#183728] hover:text-green-700 mr-4 disabled:text-gray-400"
                            >
                              {isReprocessing && selectedFile === file ? 'Processing...' : 'Reprocess'}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFile(file);
                                setFileToDelete(file);
                              }}
                              className="text-red-600 hover:text-red-900"
                            >
                              Delete Content
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={2} className="px-6 py-4 text-center text-sm text-gray-500">
                          No PDF files found in the system
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* URL management page */}
      {activeTab === 'url' && (
        <div>
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-semibold text-[#183728] mb-4">Add URL to Database</h2>
            <div className="mb-4">
              <label htmlFor="url-input" className="block text-sm font-medium text-gray-700 mb-1">
                Enter URL to add to knowledge base
              </label>
              <div className="flex">
                <input
                  type="url"
                  id="url-input"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="flex-1 min-w-0 block w-full px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-green-500 focus:border-green-500"
                />
                <button
                  onClick={() => extractUrl(url)}
                  disabled={!url || isExtracting}
                  className="px-4 py-2 bg-[#183728] text-white rounded-r-md hover:bg-green-700 focus:outline-none disabled:opacity-50"
                >
                  {isExtracting ? 'Extracting...' : 'Extract Content'}
                </button>
      </div>
    </div>
            
            {urlPreview && (
              <div className="mt-4 p-4 border border-gray-300 rounded-lg bg-gray-50">
                <h3 className="text-lg font-medium text-[#183728]">{urlPreview.title || 'URL Content Preview'}</h3>
                <p className="text-sm text-gray-500 mb-2">{urlPreview.url}</p>
                
                <div className="bg-white p-3 border border-gray-200 rounded-md max-h-60 overflow-y-auto mb-4">
                  <p className="text-sm text-gray-700">{urlPreview.content?.substring(0, 300)}...</p>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">Content size: {urlPreview.content?.length} characters</span>
                  <button
                    onClick={() => addUrlToIndex(url)}
                    disabled={isAddingUrl}
                    className="px-4 py-2 bg-[#7FCD89] text-[#183728] rounded-md hover:bg-green-400 focus:outline-none disabled:opacity-50"
                  >
                    {isAddingUrl ? 'Adding...' : 'Add to Database'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Delete confirmation dialog */}
      {fileToDelete && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>
            
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
            
            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              <div>
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100">
                  <svg className="h-6 w-6 text-red-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="mt-3 text-center sm:mt-5">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    Delete PDF Content
                  </h3>
                  <div className="mt-2">
                    <p className="text-sm text-gray-500">
                      Are you sure you want to delete all content of <strong>{fileToDelete}</strong> from the vector database? This action cannot be undone.
                    </p>
                  </div>
                </div>
              </div>
              <div className="mt-5 sm:mt-6 sm:grid sm:grid-cols-2 sm:gap-3 sm:grid-flow-row-dense">
                <button
                  type="button"
                  onClick={() => deletePDFContent(fileToDelete)}
                  disabled={isDeleting}
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-red-600 text-base font-medium text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 sm:col-start-2 sm:text-sm disabled:bg-red-300"
                >
                  {isDeleting ? 'Deleting...' : 'Delete'}
                </button>
                <button
                  type="button"
                  onClick={() => setFileToDelete('')}
                  className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#183728] sm:mt-0 sm:col-start-1 sm:text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
