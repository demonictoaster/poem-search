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
    const response = await axios.post('http://127.0.0.1:8000/predict', { query });
    setData(response.data)
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="name">Give me a poem about... </label>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)} />
          <button type="submit">Submit</button>
      </form>
      <Display result={data} />
    </div>
  )
}

export default PoemSearch