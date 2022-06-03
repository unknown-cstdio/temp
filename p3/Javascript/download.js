const firebaseConfig = {
    apiKey: "AIzaSyDgOQIJswEfH56KFvVAdq8KXZstPyZB3kg",
    authDomain: "hackriceprojecteating.firebaseapp.com",
    databaseURL: "https://hackriceprojecteating-default-rtdb.firebaseio.com",
    projectId: "hackriceprojecteating",
    storageBucket: "hackriceprojecteating.appspot.com",
    messagingSenderId: "371483673471",
    appId: "1:371483673471:web:169be295bac547daac904a",
    measurementId: "G-44KBRM0SXN"
  };
  firebase.initializeApp(firebaseConfig);
  
  var database = firebase.database();