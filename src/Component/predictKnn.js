import React, { useState } from "react";

function PredictKnn(props) {
  const [result, setResult] = useState("");
  const [venueCode, setVenueCode] = useState("");
  const [hour, setHour] = useState("");
  const [dayCode, setDayCode] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    const url = `http://127.0.0.1:8000/predictKNearestNeighbor?team_code=${props.homeTeam}&opp_code=${props.awayTeam}&venue_code=${parseInt(venueCode)}&hour=${hour}&day_code=${dayCode}`;
    const response = await fetch(url);
    const data = await response.json();
    setResult(data);
  };

  return (
    <div className="main">
      <form onSubmit={handleSubmit}>
        <label>
          Venue :
          <select className="sel-venue"
            value={venueCode}
            onChange={(event) => setVenueCode(event.target.value)}
          >
            <option value="">Select Venue</option>
            <option value="0">Home</option>
            <option value="1">Away</option>
          </select>
        </label>
        <label>
          Hour:
          <input className="input-hour"
            type="number"
            value={hour}
            onChange={(event) => setHour(event.target.value)}
          />
        </label>
        <label>
          Day :
          <select
            value={dayCode}
            onChange={(event) => setDayCode(event.target.value)}
          >
            <option value="">Select Day </option>
            <option value="1">Sunday</option>
            <option value="2">Monday</option>
            <option value="3">Tuesday</option>
            <option value="4">Wednesday</option>
            <option value="5">Thursday</option>
            <option value="6">Friday</option>
            <option value="7">Saturday</option>
          </select>
        </label>
        <button type="submit">Predict KNN</button>
      </form>
      <div>
      {typeof result === "object" && (

   <div>
    <h3 className="result">{props.teamName} : {result.result ? "win" : "loss"}</h3>
 </div>
)}

    </div>

    </div>
  );
}

export default PredictKnn;
