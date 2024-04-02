import React, { useState } from "react";
import "./home.css";
import topBarImage from "../../images/disaster.png";
import searchIcon from "../../images/searchIcon.png";
import { TextField } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { searchText, apiResult } from "../../redux/actions";

function Home() {
  let navigate = useNavigate();
  const [search, setSearchText] = useState("");
  const dispatch = useDispatch();

  // parse the input string from the search text field
  // check what kind of filter user requested: for eg-> state: CA -> this means user needs disasters from state california
  // so send api request for data in format CA null null
  // format is like: State DisasterType Year
  // after parsing send the string to next page that is results page to keep info of previous search 
  // and also make api request  
  function parseSearchString(search) {
    console.log(search);
    if (search.includes("state")) {
      let state = search.split(":")[1].trim();
      let apiQuery = state + " null null";
      dispatch(searchText(apiQuery));
      apiCallWithFilter(apiQuery);
    } else if (search.includes("year")) {
      let year = search.split(":")[1].trim();
      let apiQuery = "null null " + year;
      dispatch(searchText(apiQuery));
      apiCallWithFilter(apiQuery);
    } else if (search.includes("disaster")) {
      let disaster = search.split(":")[1].trim();
      let apiQuery = "null " + disaster + " null";
      dispatch(searchText(apiQuery));
      apiCallWithFilter(apiQuery);
    }
  }


// function to make api request. Will recieve data in json but would have to parse it still because some part of json data is in string
// Also: you would have to provide your own apiUrl as the url for your api might be different
// once data is received from server go to next page (results page)
// also send received data to redux to store it globally
  async function apiCallWithFilter(apiQuery) {
    console.log("executing filter api");
    const apiUrl = "https://7f39-34-75-188-237.ngrok.io/filter/";
    let response = await fetch(apiUrl + apiQuery);
    console.log(response);
    response.json().then((data) => {
      console.log(data);
      let parseJson = JSON.parse(data["data"]);
      dispatch(apiResult(parseJson));
      navigate("/results");
    });
  }

  return (
    <div className="body">
      <div className="homeContainer">
        <div
          className="topBar"
          style={{ backgroundImage: `url(${topBarImage})` }}
        />
        <div className="row">
          <div className="titleBox">
            <div className="title">NDS Portal</div>
          </div>
        </div>
        <div className="searchRow">
          <div className="searchBox">
            <TextField
              variant="standard"
              InputProps={{
                disableUnderline: true,
                startAdornment: (
                  <div
                    className="srIcon"
                    style={{ backgroundImage: `url(${searchIcon})` }}
                  />
                ),
              }}
              value={search}
              fullWidth={true}
              placeholder="Search disaster by type, location, time.."
              onChange={(e) => {
                setSearchText(e.target.value);
              }}
            />
          </div>
        </div>
        <div className="searchButtonRow">
          <div
            className="searchButtonBox"
            onClick={() => {
              parseSearchString(search);
            }}
          >
            <div className="searchButtonText">Search</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
