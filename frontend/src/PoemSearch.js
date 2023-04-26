import React, { useState } from "react";
import axios from 'axios';

function Display(props) {
  return (
    <div>
      <pre>{props.result.title}</pre>
      <pre>{props.result.poem}</pre>
      <pre>By {props.result.author}</pre>
      <pre>[Score: {props.result.score}, time: {props.result.time}]</pre>
    </div>
  );
}

function PoemSearch() {
  const [query, setQuery] = useState('')
  const [data, setData] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()  // prevent page from reloading when submitting
    const API_URL = process.env.REACT_APP_API_URL;
    const response = await axios.post(API_URL, { query });
    setData(response.data)
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="name">Give me a poem about... </label>
        <br></br>
        <textarea
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <br></br>
        <button type="submit">Search</button>
      </form>
      {data && (
        <Display result={data} />
      )}
    </div>
  )
}

export default PoemSearch