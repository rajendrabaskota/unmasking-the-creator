import React, { useState } from 'react';
import Select from 'react-select';
import axios from 'axios';

import { defaultText } from './utils';
import './App.css';

const baseUrl = "http://localhost:8000";

function App() {

  const selectTextDetectionOptionList = [
    { value: "direct", label: "Direct classification" },
    { value: "perplexity", label: "Using perplexity score" },
    { value: "domain_classification", label: "Using domain-wise perplexity score" },
  ]

  const selectContentTypeList = [
    { value: "TEXT_DETECTION", label: "AI-generated Text Detection" },
    { value: "IMAGE_DETECTION", label: "AI-generated Image Detection" },
  ]

  const [selectedContentType, setSelectedContentType] = useState(selectContentTypeList[0])

  const [inputText, setInputText] = useState('')
  const [pickedImage, setPickedImage] = useState(null)

  // selectedMethod = one of the items in selectTextDetectionOptionList
  // use selectedMethod.value
  const [selectedMethod, setSelectedMethod] = useState(selectTextDetectionOptionList[0])
  // output should be like: {creator: "AI", probability: 0.93241874}
  const [output, setOutput] = useState({})
  const [checking, setChecking] = useState(false)
  // for showing or not showing empty result UI at first
  const [showResultUI, setShowResultUI] = useState(true)

  const onImageChange = (e) => {
    let img = e.target.files[0];
    setPickedImage(img)
  }

  const analyzeText = async () => {
    if(!inputText) {
      alert("Please enter the text");
      return;
    }

    setChecking(true);

    axios.post(`${baseUrl}/analyze-text`, {
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

  const analyzeImage = async () => {
    if(!pickedImage) {
      alert("Please upload an image");
      return;
    }

    setChecking(true);

    const formData = new FormData();
    formData.append('file', pickedImage);

    axios.post(`${baseUrl}/analyze-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
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


      /***
        import os

        # Define a directory to save the uploaded files
        UPLOAD_DIR = "uploads"

        # Ensure the upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        @app.post("/analyze-image")
        async def upload_image(file: UploadFile = File(...)):
            try:
                # Save the uploaded file to disk
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                with open(file_path, "wb") as f:
                    f.write(file.file.read())

                return {"message": "SUCCESS", "filename": file.filename, "file_path": file_path}
            except Exception as e:
                return {"message": "FAILURE", "filename": file.filename, "file_path": file_path}

       * ***/
  }

  return (
    <div className="App">
      <nav className="App-Navbar">
        <h1 className="App-Navbar-title">Unmasking the Creator</h1>
      </nav>

      <Select
        className="App-content-type-select"
        options={selectContentTypeList}
        value={selectedContentType}
        onChange={(option) => setSelectedContentType(option)}
      />

      <section className="App-check-section">
        {selectedContentType.value == "TEXT_DETECTION" ?
          (
            <div className="App-input-div">
              <h2 className="App-input-title">Write the text here</h2>
              <div className="App-editor-div">
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
                options={selectTextDetectionOptionList}
                value={selectedMethod}
                onChange={(option) => setSelectedMethod(option)}
              />

              <button
                className="App-input-check-button"
                type='submit'
                onClick={() => analyzeText()}
              >
                {checking? "Checking...": "Check"}
              </button>
            </div>
          )
          :
          (
            <div className="App-input-div">
              <h2 className="App-input-title">Upload image</h2>

              {pickedImage && <div className="App-editor-div">
                <img src={URL.createObjectURL(pickedImage)} />
              </div>}

              <input
                className="App-input-image-picker"
                type="file"
                name="myImage"
                accept="image/*"
                onChange={onImageChange}
              />

              <button
                className="App-input-check-button"
                type='submit'
                onClick={() => analyzeImage()}
              >
                {checking? "Checking...": "Check"}
              </button>
            </div>
          )
        }

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

            <div className="App-output-text-div">
              {
              Object.keys(output).length !== 0 && output.method === "image" &&
              <p className="App-output-text">
                <p className="App-output-text">
                  {/* output.creator = "AI" or "Human" */}
                  {/* 0 <= output.probability <= 1 */}
                  This image is {output.creator}
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
