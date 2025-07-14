import axios from 'axios';

const API_URL = '/api/v1/dashboard';

export const getOverviewMetrics = async () => {
  const response = await axios.get(`${API_URL}/overview`);
  return response.data;
};
