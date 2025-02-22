/*<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script type="module" src="{% static 'js/chartSetup.js' %}"></script>
#chart{
    margin: 30px auto 30px;
    width:100%;
     <canvas id="chart" height="45px"></canvas>
}*/
document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found.');
        return;
    }
    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, -10, 0, 100);
    gradient.addColorStop(0, 'rgba(250,0,0,1)');
    gradient.addColorStop(1, 'rgba(136,255,0,1)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time').textContent;
        const temp = item.querySelector('.forecast-temperatureValue').textContent;
        const hum = item.querySelector('.forecast-humidityValue').textContent;

        if (time && temp && hum) {
            times.push(time);
            temps.push(temp);
        }
    });
    //ensure all values are valid

    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing');
        return;
    }
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
});