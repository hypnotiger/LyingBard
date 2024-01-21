// Helper for displaying status messages.
const addMessage = (message) => {
	console.log(message)
	// const messagesDiv = document.querySelector('#messages');
	// messagesDiv.style.display = 'block';
	// const messageWithLinks = addDashboardLinks(message);
	// messagesDiv.innerHTML += `> ${messageWithLinks}<br>`;
	// console.log(`Debug: ${message}`);
  };
  
  // Adds links for known Stripe objects to the Stripe dashboard.
//   const addDashboardLinks = (message) => {
// 	const piDashboardBase = 'https://dashboard.stripe.com/test/payments';
// 	return message.replace(
// 	  /(pi_(\S*)\b)/g,
// 	  `<a href="${piDashboardBase}/$1" target="_blank">$1</a>`
// 	);
//   };

const appearance = {
	theme: 'stripe',

	variables: {
		fontFamily: 'Ideal Sans, system-ui, sans-serif',
		colorDanger: '#8b0000',
		spacingUnit: '2px',
		borderRadius: '4px',
		fontSizeBase: '16pt',
	},
	rules: {
		'.Label': {
			color: '#020100',
		}
	}
};

var stripe;
var amountForm;
var paymentForm;
var prevAmount;
function preventDefault(e) {
	console.log("c")
	e.preventDefault()
}
function disablePayment() {
	button = paymentForm.querySelector('button');
	button.disabled = true;
	button.style.visibility = "hidden";
	throbber = paymentForm.querySelector('.ellipsis-throbber');
	throbber.style.visibility = "visible";
}
function enablePayment() {
	button = paymentForm.querySelector('button');
	button.disabled = false;
	button.style.visibility = "visible";
	throbber = paymentForm.querySelector('.ellipsis-throbber');
	throbber.style.visibility = "hidden";
}
document.addEventListener('DOMContentLoaded', async () => {
	// Load the publishable key from the server. The publishable key
	// is set in your .env file.
	const {publishableKey} = await fetch('/donations/config').then((r) => r.json());
	if (!publishableKey) {
	  addMessage(
		'No publishable key returned from the server. Please check `.env` and try again'
	  );
	  alert('Please set your Stripe publishable API key in the .env file');
	}
  
	stripe = Stripe(publishableKey, {
	  apiVersion: '2020-08-27',
	});

	amountForm = document.getElementById('amount-form');
	amountInput = amountForm.querySelector('#amount');
	paymentForm = document.getElementById('payment-form');
	amountInput.addEventListener("focus", disablePayment)
	prevAmount = amountInput.value;
	amountInput.addEventListener("blur", (event) => {
		value = amountInput.value;
		if (isNaN(value)) {
			value = 1.00;
		}
		value = Math.max(0.5, value).toFixed(2);
		amountInput.value = value;
		if (prevAmount == value) {
			enablePayment();
		}
		else {
			updatePaymentIntent();
		}
		prevAmount = value;
	})
});
async function updatePaymentIntent() { 
	// Prevent payment before amount is set
	disablePayment();
	// We create a PaymentIntent on the server with the amount specified so that we have its clientSecret to
	// initialize the instance of Elements below. The PaymentIntent settings configure which payment
	// method types to display in the PaymentElement.
	amountFormData = new FormData(amountForm);
	amount = amountFormData.get("amount") * 100
	amountFormData.set("amount", amount); // change from dollars to cents (Stripe works in cents)
	const {
	  error: backendError,
	  clientSecret: clientSecret,
	  amount: amountCheck
	} = await fetch('/donations/payment-intent', {method: "POST", body: amountFormData}).then(r => r.json());

	if (backendError) {
		addMessage(backendError.message);
		disablePayment();
		return;
	}
	addMessage(`Client secret returned.`);
	if (amount !== Number(amountCheck)) {
		addMessage(`Amount communicated back from server (${amountCheck}) is not the amount specified by user (${amount}).`);
		disablePayment();
		return;
	}
  
	// Initialize Stripe Elements with the PaymentIntent's clientSecret,
	// then mount the payment element.
	const elements = stripe.elements({ clientSecret, appearance });
	const paymentElement = elements.create('payment');
	paymentElement.mount('#payment-element');
	// Create and mount the linkAuthentication Element to enable autofilling customer payment details
	const linkAuthenticationElement = elements.create("linkAuthentication");
	linkAuthenticationElement.mount("#link-authentication-element");
	enablePayment();
	// If the customer's email is known when the page is loaded, you can
	// pass the email to the linkAuthenticationElement on mount:
	//
	//   linkAuthenticationElement.mount("#link-authentication-element",  {
	//     defaultValues: {
	//       email: 'jenny.rosen@example.com',
	//     }
	//   })
	// If you need access to the email address entered:
	//
	//  linkAuthenticationElement.on('change', (event) => {
	//    const email = event.value.email;
	//    console.log({ email });
	//  })
  
	// When the form is submitted...
	let submitted = false;
	paymentForm.addEventListener('submit', async (e) => {
	  e.preventDefault();
  
	  // Disable double submission of the form
	  if(submitted) { return; }
	  submitted = true;
	  disablePayment();
  
	  const nameInput = document.querySelector('#name');
  
	  // Confirm the payment given the clientSecret
	  // from the payment intent that was just created on
	  // the server.
	  const {error: stripeError} = await stripe.confirmPayment({
		elements,
		confirmParams: {
		  return_url: `${window.location.origin}/donations/finish`,
		}
	  });
  
	  if (stripeError) {
		addMessage(stripeError.message);
  
		// reenable the form.
		submitted = false;
		enablePayment();
		return;
	  }
	});
  }