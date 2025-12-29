const btn = document.getElementById("diagnoseBtn");
const symptomsInput = document.getElementById("symptoms");
const loading = document.getElementById("loading");
const resultDiv = document.getElementById("result");

btn.addEventListener("click", async () => {
    const symptoms = symptomsInput.value.trim();
    if (!symptoms) {
        alert("Please enter your symptoms.");
        return;
    }

    loading.classList.remove("d-none");
    resultDiv.innerHTML = "";

    try {
        const response = await fetch("/api/diagnose", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({symptoms})
        });

        const data = await response.json();
        loading.classList.add("d-none");

        if (data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        renderResult(data);

    } catch (err) {
        loading.classList.add("d-none");
        resultDiv.innerHTML =
            `<div class="alert alert-danger">Server error. Try again later.</div>`;
    }
});

function renderResult(data) {
    let html = `
    <div class="card shadow-sm">
        <div class="card-body">
            <h4 class="text-success">Most Likely Condition</h4>
            <h5>${data.top_prediction.condition}</h5>
            <p><strong>Probability:</strong> ${(data.top_prediction.probability * 100).toFixed(1)}%</p>

            <p>${data.top_prediction.description}</p>
    `;

    if (data.top_prediction.precautions.length) {
        html += `<h6>Precautions</h6><ul>`;
        data.top_prediction.precautions.forEach(p => {
            html += `<li>${p}</li>`;
        });
        html += `</ul>`;
    }

    if (data.confidence_warning) {
        html += `<div class="alert alert-warning">${data.confidence_warning}</div>`;
    }

    html += `
        <a href="/download_pdf" class="btn btn-outline-primary mt-2">
            Download PDF Report
        </a>
        </div>
    </div>
    `;

    resultDiv.innerHTML = html;
}
