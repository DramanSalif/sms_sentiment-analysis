document.addEventListener("DOMContentLoaded", function () {
  // Function to send a request to the Flask API
  function analyzeSentiment(text) {
    // Construct the API endpoint
    var apiEndpoint = "/predict"; // Modify if your endpoint is different
    // Use fetch API to post the SMS text data
    fetch(apiEndpoint, {
      method: "POST",
      body: JSON.stringify({ text: text }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response and display the result
        displayResult(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  // Function to display the result in the HTML page
  function displayResult(data) {
    // Assuming 'data' contains the sentiment result. Modify as per your API response structure.
    var resultElement = document.getElementById("result");
    resultElement.innerHTML = `
      <h5>Sentiment Analysis Result:</h5>
      <p>Sentiment: ${data.sentiment}</p> 
      <p>Confidence: ${data.confidence}</p>`;
  }

  // Form submission handler
  var form = document.getElementById("analysisForm");
  form.addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent default form submission event
    var smsText = document.getElementById("smsInput").value;
    analyzeSentiment(smsText); // Send the text for analysis
  });
});
