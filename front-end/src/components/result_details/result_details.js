import React from "react";
import "./result_details.css";
import { useSelector, useDispatch } from "react-redux";
import { states } from "../results/state";
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

// most funtions are similar in result_details
// just showing couple of different data of the disaster selected by user
// in future we can display more data in this page if the dataset has more features.
// also can display results of machine learning models
// Well we wanted to do alot things for this page. But were not able to do it because of time constraint
function ResultDetails() {
  const resultModel = useSelector((state) => state.searchModelDetails);
  console.log(resultModel);
  function parseDateShow(dateInput) {
    var date = new Date(dateInput.split("T")[0]);
    return date.toDateString();
  }
  function boolString(numInput) {
    if (numInput == "0") {
      return "No";
    } else {
      return "Yes";
    }
  }
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
  return (
    <div className="resultDetailsBox">
      <div className="column">
        <div className="rowResultDetails">
          <div
            className="disasterDetailsBoxImage"
            style={{
              backgroundImage: `url(${getImageBack(
                resultModel["incident_type"]
              )})`,
            }}
          />
          <div className="column">
            <div className="disasterDetailsText2">
              Disaster Type: {resultModel["declaration_title"]}
            </div>
            <div className="disasterDetailsText2">
              Date: {parseDateShow(resultModel["declaration_date"])}
            </div>
            <div className="disasterDetailsText2">
              State: {states[resultModel["state"]]["name"]}
            </div>
            <div className="disasterDetailsText2">
              Fema Declaration: {resultModel["fema_declaration_string"]}
            </div>
          </div>
        </div>
        <div className="column">
          <div className="disasterDetailsText3">
            Individuals and Households program:{" "}
            {boolString(resultModel["ih_program_declared"])}
          </div>
          <div className="disasterDetailsText3">
            Individual Assistance program:{" "}
            {boolString(resultModel["ia_program_declared"])}
          </div>
          <div className="disasterDetailsText3">
            Public Assistance program:{" "}
            {boolString(resultModel["pa_program_declared"])}
          </div>
          <div className="disasterDetailsText3">
            Hazard Mitigation program:{" "}
            {boolString(resultModel["hm_program_declared"])}
          </div>
          <div className="disasterDetailsText3">
            Declaration request number:{" "}
            {resultModel["declaration_request_number"]}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultDetails;
