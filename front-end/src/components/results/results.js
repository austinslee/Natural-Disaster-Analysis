import "./results.css";
import topBarImage from "../../images/disaster.png";
import visualizeIcon from "../../images/visualize.png";
import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  searchModelDetails,
  visualData,
  searchText,
  apiResult,
} from "../../redux/actions";
import { addDataToMap } from "kepler.gl/actions";
import { useNavigate } from "react-router-dom";
import { TextField } from "@mui/material";
import { states } from "./state";
import {
  Earthquake,
  Biological,
  Chemical,
  CoastalStorm,
  Dam,
  Drought,
  Fire,
  FishingLosses,
  Flood,
  Freezing,
  HumanCause,
  Hurricane,
  IceStorm,
  Landslide,
  Other,
  Snow,
  Storm,
  Terrorist,
  Tornado,
  Toxic,
  Tsunami,
  Typhoon,
  Volcano,
} from "../../images";

// above alot of files are maintained locally like image files and state full names and longitude and latitude values
// would have store this information on server. But because of limited time to develop project, I kept locally
// state full name and longitude and latitude needed for visualization, As lat long values will help to place markers on map
// have kept lat long values for state's capital
function Results() {
  let navigate = useNavigate();
  const dispatch = useDispatch();
  const [filterText, setFilterText] = useState("");
  const [filterCategory, setFilterCategory] = useState("");
  // get the previous search text from last page
  const searchTextBefore = useSelector((state) => state.searchText);
  // get the api data made previously. Also will be updated automatically when calling filter api on this page
  const apiResultData = useSelector((state) => state.apiResult);
  // to keep track of disasters count per state.
  // using this value for visualization of data
  let stateResultCount = {};
  let data = apiResultData;
  // data 2 is just for keeping reference of how our visualization data format needs to be like
  // in our code we update the rows values with the data obtained from server/API
  let data2 = {
    fields: [
      {
        name: "tpep_pickup_datetime",
        format: "YYYY-M-D H:m:s",
        type: "timestamp",
      },
      { name: "pickup_longitude", format: "", type: "real" },
      { name: "pickup_latitude", format: "", type: "real" },
      { name: "intensity", format: "", type: "int" },
    ],
    rows: [
      ["2015-01-15 19:05:39", -73.99389648, 40.75011063],
      ["2015-01-15 19:05:39", -73.97642517, 40.73981094],
      ["2015-01-15 19:05:40", -73.96870422, 40.75424576],
    ],
  };
  console.log(searchTextBefore);
  console.log(filterCategory);
  let filterDropName = "Categories";
  if (filterCategory !== "") {
    filterDropName = filterCategory;
  }

  // function to parse date as date in the dataset is not easy to read
  function parseDate(dateInput) {
    //1953-05-02T00:00:00Z
    var date = new Date(dateInput.split("T")[0]);
    return (
      [
        date.getFullYear(),
        padTo2Digits(date.getMonth() + 1),
        padTo2Digits(date.getDate()),
      ].join("-") +
      " " +
      [
        padTo2Digits(date.getHours()),
        padTo2Digits(date.getMinutes()),
        padTo2Digits(date.getSeconds()),
      ].join(":")
    );
  }

  function parseDateShow(dateInput) {
    var date = new Date(dateInput.split("T")[0]);
    return date.toDateString();
  }

  function padTo2Digits(num) {
    return num.toString().padStart(2, "0");
  }

  function parseApiJson(jsonString) {
    let parseJson = JSON.parse(jsonString);
    return parseJson;
  }

  // function to make filter api call
  async function apiCallWithFilter(apiQuery) {
    const apiUrl = "https://7f39-34-75-188-237.ngrok.io/filter/";
    let response = await fetch(apiUrl + apiQuery);
    console.log(response);
    response.json().then((data) => {
      console.log(data);
      if (data["data"].length > 1) {
        let parseJson = JSON.parse(data["data"]);
        dispatch(searchText(apiQuery));
        dispatch(apiResult(parseJson));
      } else {
        // to handle if there are no results on dataset for that particular filter
      }
    });
  }

  // again parse filter text field input for api call
  function applyFilter() {
    let filters = [];
    let searchQuery = searchTextBefore.split(" ");
    filters[0] = searchQuery[0];
    filters[1] = searchQuery[1];
    filters[2] = searchQuery[2];
    if (filterCategory === "State") {
      filters[0] = filterText.trim();
    } else if (filterCategory === "DisasterType") {
      filters[1] = filterText.trim();
    } else {
      filters[2] = filterText.trim();
    }
    apiCallWithFilter(filters[0] + " " + filters[1] + " " + filters[2]);
  }

  function returnStateName(stateCode) {
    if (states[stateCode] == undefined) {
      return stateCode;
    } else {
      return states[stateCode]["name"];
    }
  }

  // get the image from local storage to given disaster types
  function getImageBack(disasterType) {
    switch (disasterType) {
      case "Earthquake":
        return Earthquake;
      case "Tornado":
        return Tornado;
      case "Flood":
        return Flood;
      case "Fire":
        return Fire;
      case "Other":
        return Other;
      case "Hurricane":
        return Hurricane;
      case "Volcano":
        return Volcano;
      case "Severe Storm(s)":
        return Storm;
      case "Toxic Substances":
        return Toxic;
      case "Typhoon":
        return Typhoon;
      case "Drought":
        return Drought;
      case "Dam/Levee Break":
        return Dam;
      case "Snow":
        return Snow;
      case "Severe Ice Storm":
        return IceStorm;
      case "Freezing":
        return Freezing;
      case "Coastal Storm":
        return CoastalStorm;
      case "Fishing Losses":
        return FishingLosses;
      case "Mud/Landslide":
        return Landslide;
      case "Human Cause":
        return HumanCause;
      case "Terrorist":
        return Terrorist;
      case "Tsunami":
        return Tsunami;
      case "Chemical":
        return Chemical;
      case "Biological":
        return Biological;
      default:
        return Other;
    }
  }

  console.log(`url(${getImageBack("Earthquake")})`);
  console.log(parseDate("1953-05-02T00:00:00Z"));

  return (
    <div className="resultsBody">
      <div className="resultsContainer">
        <div
          className="resultsTopBar"
          style={{ backgroundImage: `url(${topBarImage})` }}
        />
        <div className="resultsRow">
          <div className="titleBox">
            <div className="title">NDS Portal</div>
          </div>
        </div>
        <div className="centerRow">
          <div className="searcResultsTextRow">
            <div className="searchResultsText">Search results</div>
          </div>
          <div
            className="visualizationButtonBox"
            onClick={() => {
              // add data to view in kepler.gl
              // have to update rows data for vizualization
              let rows = [];
              for (let index = 0; index < data.length; index++) {
                let count = 0;
                if (stateResultCount[data[index]["state"]] === undefined) {
                  count = 1;
                  stateResultCount[data[index]["state"]] = 1;
                } else {
                  count = stateResultCount[data[index]["state"]];
                  stateResultCount[data[index]["state"]] = count + 1;
                }
                if (states[data[index]["state"]] != undefined) {
                  rows.push([
                    parseDate(data[index]["declaration_date"]),
                    states[data[index]["state"]]["longitude"],
                    states[data[index]["state"]]["latitude"],
                    count,
                  ]);
                }
              }
              data2.rows = rows;
              // send redux the updated vizualization data, so that map loads the updated data
              dispatch(
                visualData({
                  datasets: {
                    info: {
                      label: "NaturalDisaster",
                      id: "NaturalDisaster",
                    },
                    data: data2,
                  },
                  option: {
                    centerMap: true,
                    readOnly: false,
                  },
                  config: {},
                })
              );
              navigate("/visualization");
            }}
          >
            <div
              className="visualizationButtonIcon"
              style={{ backgroundImage: `url(${visualizeIcon})` }}
            />
          </div>
        </div>
        <div className="filterRow">
          <div class="dropdown">
            <button class="dropbtn">{filterDropName}</button>
            <div class="dropdown-content">
              <a
                href="#"
                onClick={() => {
                  setFilterCategory("State");
                }}
              >
                State
              </a>
              <a
                href="#"
                onClick={() => {
                  setFilterCategory("DisasterType");
                }}
              >
                Disaster Type
              </a>
              <a
                href="#"
                onClick={() => {
                  setFilterCategory("Year");
                }}
              >
                Year
              </a>
            </div>
          </div>
          <div className="filterBox">
            <TextField
              variant="standard"
              InputProps={{
                disableUnderline: false,
              }}
              value={filterText}
              fullWidth={true}
              placeholder="Filter text"
              onChange={(e) => {
                setFilterText(e.target.value);
              }}
            />
          </div>
          <div
            className="searchButtonBox"
            onClick={() => {
              // filter api call ::::::::::::::::::::::::::::::::::::::::::
              applyFilter();
            }}
          >
            <div className="filterButtonText">Filter</div>
          </div>
        </div>
        <div className="TotalEnteries">
          <h1>Total enteries: {data.length}</h1>
        </div>
        <div className="wrapper">
          {data.map((disaster, index) => {
            return (
              <div key={index}>
                <div className="disastersResultsBoxRow">
                  <div
                    className="disastersResultsBox"
                    key={disaster["declaration_request_number"]}
                    onClick={() => {
                      dispatch(searchModelDetails(disaster));
                      navigate("/resultsDetails");
                    }}
                  >
                    <div className="disastersBoxColumn">
                      <div className="disasterDetailsText">
                        Disaster Type: {disaster["incident_type"]}
                      </div>
                      <div className="disasterDetailsText">
                        Disaster declaration: {disaster["declaration_title"]}
                      </div>
                      <div className="disasterDetailsText">
                        Date: {parseDateShow(disaster["declaration_date"])}
                      </div>
                      <div className="disasterDetailsText">
                        State: {returnStateName(disaster["state"])}
                      </div>
                      <div className="disasterDetailsText">
                        Fema Declaration: {disaster["fema_declaration_string"]}
                      </div>
                    </div>
                    <div
                      className="disasterBoxImage"
                      style={{
                        backgroundImage: `url(${getImageBack(
                          disaster["incident_type"]
                        )})`,
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default Results;
