document.addEventListener("DOMContentLoaded", function () {
  var form = document.getElementById("analysisForm");
  form.onsubmit = function (e) {
    e.preventDefault(); // to prevent the default form submission
    var smsText = document.getElementById("smsInput").value;

    // Perform the AJAX request
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: smsText }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display the result
        document.getElementById("result").textContent =
          "Sentiment: " + data.prediction;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };
});
