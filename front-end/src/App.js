import "./App.css";
import React from "react";
import { useSelector, useDispatch } from "react-redux";
import { increment } from "./redux/actions";
import Home from "./components/home/home";

function App() {
  const counter = useSelector((state) => state.counter);
  const dispatch = useDispatch();
  return (
    <div className="App">
      {/* <h1>Counter:{counter}</h1>
      <button
        onClick={() => {
          dispatch(increment());
        }}
      >
        +
      </button> */}
      <Home />
    </div>
  );
}

export default App;
