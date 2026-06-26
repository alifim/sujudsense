// public/analytics.js
const script = document.createElement('script');
script.src = "https://www.googletagmanager.com/gtag/js?id=G-Q3KJ8GSC8F";
script.async = true;
document.head.appendChild(script);

window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

// Replace with your actual G-XXXXXXXXXX ID
gtag('config', 'G-Q3KJ8GSC8F');