import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import { applyMiddleware, compose, createStore } from "redux";
import { taskMiddleware } from "react-palm/tasks";
import Results from "./components/results/results";
import ResultDetails from "./components/result_details/result_details";
import allReducers from "./redux/reducers";
import { Provider } from "react-redux";
import Visualization from "./components/visualization/visualization";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

const initialState = {};
const enhancers = [applyMiddleware(taskMiddleware)];
let store = createStore(allReducers, initialState, compose(...enhancers));

ReactDOM.render(
  <Provider store={store}>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/results" element={<Results />} />
        <Route path="/resultsDetails" element={<ResultDetails />} />
        <Route path="/visualization" element={<Visualization />} />
      </Routes>
    </Router>
  </Provider>,

  document.getElementById("root")
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
