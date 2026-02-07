import React, { useEffect, useState } from 'react';
import DashboardLayout from '../../layout/dashboardlayout';
import Sidebar from '../sidebar/sidebar';
import { SimulationParams, SimulationResult } from '../../types';
import BellCurve from './BellCurve';

export default function Dashboard() {
  // 1. Manage the inputs from the Sidebar
  const [params, setParams] = useState<SimulationParams>({
    predictionType: 'Seat',
    coalition: 'BN',
    baseVoterPercent: 50,
    swingPercent: 0,
    turnoutRate: 85,
    specificValue: ''
  });

  // 2. Manage the data returned from the "Backend"
  const [result, setResult] = useState<SimulationResult | null>(null);
  
  // 3. Track if the AI is currently "thinking"
  const [isLoading, setIsLoading] = useState(false);

  // This state holds the data that will eventually come from your CSV
  const [filterLists, setFilterLists] = useState({
    states: ['Selangor', 'Johor', 'Penang', 'Perak', 'Kedah'], 
    regions: ['Central', 'Northern', 'Southern', 'East Coast', 'Sabah', 'Sarawak'], 
    seats: [] 
  });

  // This runs once when the dashboard opens
    useEffect(() => {
        const fetchOptions = async () => {
            try {
            // Calling your new FastAPI endpoint
                console.log("Attempting to fetch from Python...");
                const response = await fetch('http://127.0.0.1:8000/api/options'); // Use 127.0.0.1 instead of localhost
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                        }
            
            const data = await response.json();
            console.log("Data received from Backend:", data);
            
            // This automatically updates the sidebar with real GE14 data!
            setFilterLists({
                states: data.states || [],
                regions: data.regions || [],
                seats: data.seats || []
            });

            } catch (error) {
                console.error("The bridge is broken! Error:", error);
                }
            
        };

        fetchOptions();
    }, []);


  // 4. The Synchronization Logic (The "Handshake")
  const handleRunSimulation = async () => {
    setIsLoading(true); // Start the loading state
    
    // Simulating a network request to your Python/AI backend
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seat_id: params.specificValue.split(' ')[0], // Extracts 'P106'
          state: "Unknown", // Pipeline will fix this
          coalition: params.coalition,
          total_registered: 150000, 
          turnout_rate: params.turnoutRate,    // Matches Python SeatInput
          base_support: params.baseVoterPercent, // Syncing 'baseVoterPercent' to 'base_support'
          swing: params.swingPercent
        })
      });
      const data = await response.json();
      setResult({
        coalition: params.coalition,
        predictionType: params.predictionType,
        result: `Verdict: ${data.verdict}`,
        probability: Math.round(data.probability * 100)
      });
    } catch (error) {
      console.error("Error fetching prediction:", error);
    } finally {
      setIsLoading(false); // End the loading state
    }
  };
  
  return (
    <DashboardLayout>
      {/* SIDEBAR: Passes state and the 'Run' trigger to the left panel */}
      <Sidebar params={params} setParams={setParams} onRun={handleRunSimulation} options={filterLists} />
      
      {/* MAIN CONTENT: Displays synchronized data in your wireframe layout */}
      <main className="flex-1 max-w-[1600px] mx-auto p-10 space-y-8 bg-gray-50">

        <div className="w-full space-y-6">
          {/* Section: Coalition Name */}
          <section>
            <label className="font-bold block mb-1">Coalition:</label>
            <div className="border-2 border-black p-3 bg-white min-h-[50px] flex items-center">
                {isLoading ? (
                  <span className="text-gray-400 italic">Calculating...</span>
                ) : (
                  result?.coalition || 'Waiting for input...'
                )}
            </div>
          </section>

          {/* Section: Prediction Scope */}
          <section>
            <label className="font-bold block mb-1">Prediction Type:</label>
            <div className="border-2 border-black p-3 bg-white min-h-[50px] flex items-center">
                {isLoading ? '' : result?.predictionType}
            </div>
          </section>

          {/* Section: Qualitative Result */}
          <section>
            <label className="font-bold block mb-1">Result:</label>
            <div className="border-2 border-black p-3 bg-white w-2/3 min-h-[50px] flex items-center">
                {isLoading ? '' : result?.result}
            </div>
          </section>

          {/* Section: Probability & Data Viz */}
          <div className="flex gap-8 items-end">
            <div className="w-32">
              <label className="font-bold block mb-1">Probability:</label>
              <div className="border-2 border-black p-3 bg-white text-center font-mono text-xl">
                {isLoading ? '--%' : (result ? `${result.probability}%` : '0%')}
              </div>
            </div>
            
            <div className="flex-1">
               <label className="font-bold block mb-1 text-center">Confidence Distribution</label>
               {/* This is the synchronized math visual we built */}
               <BellCurve centerPoint={result?.probability || 50} />
            </div>
          </div>
        </div>
      </main>
    </DashboardLayout>
  );
}