import React, { useState } from "react";

function PredictionComponent(props) {
  const [result, setResult] = useState("");
  const [venueCode, setVenueCode] = useState("");
  const [hour, setHour] = useState("");
  const [dayCode, setDayCode] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    const url = `http://127.0.0.1:8000/predictRandomForest?team_code=${props.homeTeam}&opp_code=${props.awayTeam}&venue_code=${parseInt(venueCode)}&hour=${hour}&day_code=${dayCode}`;
    const response = await fetch(url);
    const data = await response.json();
    setResult(data);
  };

  return (
    <div className="main">
      <form className="mainForm" onSubmit={handleSubmit}>
        <label className="lable-venue">
          Venue :
          <select className="sel-venue"
            value={venueCode}
            onChange={(event) => setVenueCode(event.target.value)}
          >
            <option className="op-venu" value="">Select Venue</option>
            <option  className="op-venu" value="0">Home</option>
            <option  className="op-venu" value="1">Away</option>
          </select>
        </label>
        <label  className="lable-hour">
          Hour:
          <input className="input-hour"
            type="number"
            value={hour}
            onChange={(event) => setHour(event.target.value)}
          />
        </label>
        <label  className="lable-day">
          Day :
          <select className="sel-day"
            value={dayCode}
            onChange={(event) => setDayCode(event.target.value)}
          >
            <option  className="op-day" value="">Select Day </option>
            <option className="op-day" value="1">Sunday</option>
            <option className="op-day" value="2">Monday</option>
            <option className="op-day" value="3">Tuesday</option>
            <option className="op-day" value="4">Wednesday</option>
            <option className="op-day" value="5">Thursday</option>
            <option className="op-day" value="6">Friday</option>
            <option className="op-day" value="7">Saturday</option>
          </select>
        </label>
        <button className="button" type="submit">Predict RF</button>
      </form>
      <div>
      {typeof result === "object" && (

<h3 className="result">{props.teamName} : {result.result ? "win" : "loss"}</h3>
)}

    </div>

    </div>
  );
}

export default PredictionComponent;
