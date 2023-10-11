import React, { useState } from 'react';
import Select from 'react-select';
import axios from 'axios';

import { defaultText } from './utils';
import './App.css';

// const baseUrl = "http://localhost:8000";  // Uncomment this line if running in your local machine
const baseUrl = "http://54.234.146.152:8000"; // BaseURL for AWS

function App() {

  const selectOptionList = [
    { value: "direct", label: "Direct classification" },
    { value: "perplexity", label: "Using perplexity score" },
    { value: "domain_classification", label: "Using domain-wise perplexity score" },
  ]

  const [inputText, setInputText] = useState('')
  // selectedMethod = one of the items in selectOptionList
  // usee selectedMethod.value (i.e. 0, 1 or 2)
  const [selectedMethod, setSelectedMethod] = useState(selectOptionList[0])
  // output should be like: {creator: "AI", probability: 0.93241874}
  const [output, setOutput] = useState({})
  const [checking, setChecking] = useState(false)
  // for showing or not showing empty result UI at first
  const [showResultUI, setShowResultUI] = useState(true)

  const analyzeCode = async () => {
    if(!inputText) {
      alert("Please enter the text");
      return;
    }

    setChecking(true);

    axios.post(`${baseUrl}/analyze`, {
        "text": inputText,
        "method": selectedMethod.value
      }, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
      })
      .then((response) => {
        console.log(response.data.result);
        const result = response.data.result;

        setOutput(result);
        setShowResultUI(true);
      })
      .catch(error => {
        alert(error.message);
      })
      .finally(() => {
        setChecking(false);
      })
  }

  return (
    <div className="App">
      <nav className="App-Navbar">
        <h1 className="App-Navbar-title">Unmasking the Creator</h1>
      </nav>
      <section className="App-text-check-section">
        <div className="App-text-input-div">
          <h2 className="App-text-input-title">Write the text here</h2>
          <div className="App-text-editor-div">
            <textarea
                name="input-text"
                type="text"
                id="input-text"
                rows="16"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                required
            />
          </div>

          <Select
            className="App-method-select"
            options={selectOptionList}
            value={selectedMethod}
            onChange={(option) => setSelectedMethod(option)}
          />

          <button
            className="App-text-input-check-button"
            type='submit'
            onClick={() => analyzeCode()}
          >
            {checking? "Checking...": "Check"}
          </button>
        </div>
        {showResultUI &&
        <div className="App-output-div" id="output">
          <h2 className="App-output-title">Result</h2>
          <div className="App-output">
            <div className="App-output-text-div">
              {
              Object.keys(output).length !== 0 && output.method === "direct" &&
              <p className="App-output-text">
                <p className="App-output-text">
                  {/* output.creator = "AI" or "Human" */}
                  {/* 0 <= output.probability <= 1 */}
                  This text is {output.creator}
                </p>
                <p className="App-output-text">
                  Probability: {output.probability}
                </p>
              </p>
              }
            </div>

            <div className="App-output-text-div">
              {
              Object.keys(output).length !== 0 && output.method === "perplexity" &&
              <p className="App-output-text">
                <p className="App-output-text">
                  {/* output.creator = "AI" or "Human" */}
                  {/* 0 <= output.probability <= 1 */}
                  This text is {output.creator}
                </p>
                <p className="App-output-text">
                  Perplexity: {output.perplexity}
                </p>
                <p className="App-output-text">
                  Perplexity Threshold: {output.threshold}
                </p>
              </p>
              }
            </div>

            <div className="App-output-text-div">
              {
              Object.keys(output).length !== 0 && output.method === "domain_classification" &&
              <p className="App-output-text">
                <p className="App-output-text">                                                                                                                                                                           
                  {/* output.creator = "AI" or "Human" */}
                  {/* 0 <= output.probability <= 1 */}
                  This text is {output.creator}
                </p>
                <p className="App-output-text">
                  Perplexity: {output.perplexity}
                </p>
                <p className="App-output-text">
                  Perplexity Threshold: {output.threshold}
                </p>
                <p className="App-output-text">
                  Domain: {output.domain}                                                                 
                </p>
              </p>
              }
            </div>                                                                                    

          </div>
        </div>
        }
      </section>
    </div>
  );
}

export default App;
