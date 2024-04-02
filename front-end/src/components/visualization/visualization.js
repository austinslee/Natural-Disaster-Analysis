import "./visualization.css";
import React from "react";
import KeplerGl from "kepler.gl";
import { useSelector, useDispatch } from "react-redux";
import { addDataToMap } from "kepler.gl/actions";
import useSwr from "swr";

// Using kepler.gl library for the map loadup
// most functions independently handled by kepler.gl, just we need to provided up-to-date data with correct values
function Visualization() {
  const dispatch = useDispatch();
  // get the vizualization data form redux
  const mapData = useSelector((state) => state.visualData);
  const finalMapData = mapData["datasets"]["data"];
  const { data } = useSwr("naturaldisasters", async () => {
    return finalMapData;
  });
  // let react know to refresh the page if updated data is given
  React.useEffect(() => {
    if (finalMapData) {
      dispatch(
        // addDataToMap is for telling the kepler.gl library to load up the data on to the map
        addDataToMap({
          datasets: {
            info: {
              label: "naturaldisasters",
              id: "naturaldisasters",
            },
            data,
          },
          // can customize options as per your need
          // check kepler.gl documentation to know more details on options you can change
          option: {
            centerMap: true,
            readOnly: false,
          },
          // you can provide config to the map to add some features/configurations by default
          // you can visit the kepler.gl documentation website page to know all syntax for providing configs
          // configs would be like: giving the circles on map a default color, changing map style, etc.
          config: {},
        })
      );
    }
  }, [dispatch, data]);

  // you would have to create a account in kepler.gl website to get the mapboxApiAccessToken
  // you can use the free account if just playing around with library or building a small project
  return (
    <div>
      <KeplerGl
        id="naturaldisaster"
        mapboxApiAccessToken={
          "add_your_token_key_here.."
        }
        width={window.innerWidth}
        height={window.innerHeight}
      />
    </div>
  );
}

export default Visualization;
