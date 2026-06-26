// public/analytics.js
console.log("Analytics injector started...");

const script = document.createElement('script');
script.src = "https://www.googletagmanager.com/gtag/js?id=G-XJPK4TFD12";
script.async = true;

script.onload = () => {
  console.log('✅ GA4 Remote script loaded successfully.');
  
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  
  // Force GA4 to track the actual URL, not the Hugging Face iframe URL
  gtag('config', 'G-XJPK4TFD12', {
      page_path: window.location.pathname,
  });
  
  console.log('✅ GA4 Configured and pageview sent.');
};

script.onerror = () => {
  console.error('❌ Error loading GA4 remote script. Check your ad-blocker!');
};

document.head.appendChild(script);