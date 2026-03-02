import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from 'recharts';

// 1. Define what this component needs to function
interface BellCurveProps {
  centerPoint: number; // This maps to your "Probability" result
}

const BellCurve: React.FC<BellCurveProps> = ({ centerPoint }) => {
  
  // 2. The Math Layer: Transform a single number into a curve
  // useMemo ensures we don't recalculate this unless the probability changes
  const chartData = useMemo(() => {
    const points = [];
    const standardDeviation = 12; // Controls how "wide" the curve is
    
    // We generate 50 points from 0 to 100 to draw a smooth line
    for (let i = 0; i <= 100; i += 2) {
      // Normal Distribution Formula (simplified)
      const exponent = -Math.pow(i - centerPoint, 2) / (2 * Math.pow(standardDeviation, 2));
      const value = Math.exp(exponent);
      
      points.push({
        percentage: i,
        probabilityDensity: value,
      });
    }
    return points;
  }, [centerPoint]);

  // 3. The Visual Layer: Render the data using Recharts
  return (
    <div className="w-full h-40 border-2 border-black bg-white p-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          {/* We hide the axes to match your clean wireframe design */}
          <XAxis dataKey="percentage" hide />
          <YAxis hide />
          
          <Area 
            type="monotone" 
            dataKey="probabilityDensity" 
            stroke="#000"      // Black line
            fill="#d1d5db"     // Gray fill (matches sketch aesthetic)
            strokeWidth={2}
            animationDuration={800} // Adds a "live" feel when results load
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BellCurve;