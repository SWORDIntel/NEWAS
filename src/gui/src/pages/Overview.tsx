import React, { useEffect, useState } from 'react';
import { getOverviewMetrics } from '../services/api';

const Overview: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      const data = await getOverviewMetrics();
      setMetrics(data);
    };

    fetchMetrics();
  }, []);

  if (!metrics) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1>Dashboard Overview</h1>
      <pre>{JSON.stringify(metrics, null, 2)}</pre>
    </div>
  );
};

export default Overview;
