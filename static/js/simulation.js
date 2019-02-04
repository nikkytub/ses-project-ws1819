let map;
let carIcon = {
    url: "static/images/car.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedCarIcon = {
    url: "static/images/selected-car.png",
    scaledSize: new google.maps.Size(30, 30)
};

let stationIcon = {
    url: "static/images/station.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedStationIcon = {
    url: "static/images/selected-station.png",
    scaledSize: new google.maps.Size(20, 20),
};
let autoDriveTimer;
let newTimer;
let routeMarkers=[];
let carMarkers=[];
let carMarker;
let stationMarkers=[];
let directionsService = new google.maps.DirectionsService;
let tuberlin = new google.maps.LatLng(52.512230, 13.327135);
let destinationIcon = 'https://chart.googleapis.com/chart?' + 'chst=d_map_pin_letter&chld=D|FF0000|000000';
let flag = true;
let gpsCoords = [];
let directionsDisplay;
let timer;
let locationsToDistination =[];
let locationsToChargingStation =[];
let myinterval2;
let routePoints = [];
let pointDuration = [];


var polyline = new google.maps.Polyline({
    path: [],
    strokeColor: '#FF0000',
    strokeWeight: 3
});

var myinterval;
$(document).ready(function() {
    initMap();
    loadGrids(gridList);
    //newwwInterval = setInterval(get_grids, 5000);
    //showCar();
    calculateAndDisplayRoute(directionsDisplay, directionsService,  new google.maps.LatLng(car.lat,car.lon), tuberlin,1);
    removeCarsMarker();
    carMarker= addMarker('car'+ car.id +'\nsoc: '+car.soc
            , new google.maps.LatLng(car.lat,car.lon),selectedCarIcon);
    carMarkers.push(carMarker);
    myinterval = setInterval(showCar, 1000);
});

function initMap(){
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 13,
        center: new google.maps.LatLng(52.5159995,13.3560001),
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });

    directionsDisplay  = new google.maps.DirectionsRenderer({map: map,polylineOptions: {
            strokeColor: "DeepSkyBlue",
            strokeWeight: 6,
            strokeOpacity: 0.8
        },suppressMarkers: true });
}


function loadCars(Auto_cars){
    for (let i = 0 ; i < Auto_cars.length ; i++)
    {
        $("#cars").append( '<option value='+Auto_cars[i].id+'>car' + Auto_cars[i].id + '</option>' );
        let carMarker= addMarker('car'+ Auto_cars[i].id +'\nsoc: '+Auto_cars[i].soc
            , new google.maps.LatLng(Auto_cars[i].lat,Auto_cars[i].lon),carIcon);

        carMarker.addListener('click', function() {
              showCar(Auto_cars[i].id);
          });
        carMarkers.push(carMarker);
    }
    $("#cars").selectpicker("refresh");

    return Auto_cars;
}
function get_grids (){
    //alert(car.soc);

    $.ajax({
                 type: "POST",
                 url: "/load_grids",
                 data: JSON.stringify(car),
                 success: function(data){
                     loadGrids(data);
        }
        ,dataType: 'json'
    });
}
function loadGrids(gridList){
    removeStationsMarker();
    for (let i = 0 ;i<gridList.length;i++)
    {

        let stationMarker =addMarker('grid'+ gridList[i].name +'\nprice: '+gridList[i].price
            +'\nalpha: '+gridList[i].alpha, new google.maps.LatLng(gridList[i].lat,gridList[i].lon),stationIcon);
        stationMarkers.push(stationMarker);
    }
    return gridList;
}

$("#cars").change(function () {
    let selectedCar= $(this).find("option:selected").val();
        showCar(selectedCar);
});

$("#stations").change(function () {
    let selectedGrid= $(this).find("option:selected").val();
        showGrid(selectedGrid);
});

function addMarker(title,position,icon) {
    return new google.maps.Marker({
        title:title,
        position: position,
        map: map,
        icon: icon
    });
}

function removeRouteMarker(){
    routeMarkers.map(function(marker){
        marker.setMap(null);
    } );
    routeMarkers = []
}

function removeStationsMarker(){
    stationMarkers.map(function(marker){
        marker.setMap(null);
    } );
    stationMarkers = []
}

function removeCarsMarker(){
    carMarkers.map(function(marker){
        marker.setMap(null);
    } );
    carMarkers = []
}

function showCar(){
    clearInterval(myinterval);
    $('#speedInput').val(car.speed);
    if(car.mode == "eco_mode"){
        $('#eco').bootstrapToggle('on');
        $('#cost').attr("disabled","");
        $('#time').attr("disabled","");
    }

    else if (car.mode == "costSaving_mode"){
        $('#cost').bootstrapToggle('on');
        $('#time').attr("disabled","");
        $('#eco').attr("disabled","");


    }
    else if (car.mode == "chargingtime_mode"){
        $('#time').bootstrapToggle('on');
        $('#eco').attr("disabled","");
        $('#cost').attr("disabled","");

    }
    $('#battery').attr('src', getBatteryIcon2(car.soc * 100));
    $('#b1soc').text(Math.floor(car.soc * 10000)/100 + '%');
    //removeCarsMarker();
    //carMarker= addMarker('car'+ car.id +'\nsoc: '+car.soc
     //       , new google.maps.LatLng(car.lat,car.lon),selectedCarIcon)
    //carMarkers.push(carMarker);
    carMarker.setPosition(new google.maps.LatLng(car.lat,car.lon));
    map.setCenter(carMarker.position);
    if(locationsToDistination.length > 0 && flag)
    {startSimulation();
    alert("simulation is starting")}
    }

function startSimulation(){
    flag = false;

    autoDriveTimer = setInterval(function () {
            // stop the timer if the route is finished
            if (locationsToDistination.length === 0) {
                clearInterval(autoDriveTimer);
            } else {

                if (car.soc > 0.2) {
                    // move marker to the next position (always the first in the array)

                    moveCar(locationsToDistination[0]);
                    // remove the processed position
                    locationsToDistination.shift();
                }
               else {

                }
            }
        },
        100);
    //for (i = 5; i<locationsToDistination.length; i++){
    //    window.setTimeout(function() {moveCar(locationsToDistination[i]);
    //    }, 5000);

    //}
}

function moveCar(newLocation){

    powerstate = car.soc*car.capacity;
    //alert("powerPD" +car.powerPD);
    consumption = 0.001*car.powerPD;
    //alert(newLocation);
    if (powerstate-consumption > 0){
        //alert("powerstate"+powerstate);
        //alert("consumption"+consumption);
        // available energy - needed energy
        powerupdate = powerstate-consumption;
        //alert("powerupdate"+powerupdate);
        // percentage battery left (new state of charge)
        soc_update = powerupdate/car.capacity;
        //soc_update = Math.floor((powerupdate/car.capacity) *100 )/100;

        //alert ("soc"+soc_update);
        if (soc_update <= 0) {
            //alert(soc_update);
            alert("Battery below 1%")
        }
        if (soc_update <= 0.2) {
            alert (" POWER IS LOW \n Searching for a charging station...");
            clearInterval(autoDriveTimer);
            get_grids();
             $.ajax({
                 type: "POST",
                 url: "/postCar_getGrid",
                 data: JSON.stringify(car),
                 success: function(data){
                     alert("The optimal grid is grid"+data.id);
                     //showGrid(data.id);
                     let gridLocation = new google.maps.LatLng(data.lat, data.lon);
                     //alert(data);
                     $("#grid").removeAttr("hidden");
                     showGrid(data);
                     //alert("gridLocation,,,," + gridLocation);
                     calculateAndDisplayRoute(directionsDisplay, directionsService,newLocation,gridLocation,0);
                     driveToCharginStation();
        }
        ,dataType: 'json'
    });
            //grid_loc = optimize(grids, modus);
            //soc = 1.0;
            //code = 1;
        }
        else
            {
            car.soc = soc_update;
            car.lat = newLocation.lat();
            car.lon = newLocation.lng();
            get_grids();
            //alert("car location"+car.lat,car.lon);
            carMarker.setPosition(newLocation);
            map.setCenter(carMarker.position);
        }
        showCar();
    }
}

function driveToCharginStation(){
    //clearInterval(myinterval2);
    newTimer = setInterval(function () {
        //alert("locationToCharging"+locationsToChargingStation.length);
            // stop the timer if the route is finished
            if (locationsToChargingStation.length === 0) {
                clearInterval(newTimer);
                car.soc = 1;
                flag = 1;
                $("#grid").attr("hidden","");
                timeOut = setTimeout(function (){
                    calculateAndDisplayRoute(directionsDisplay, directionsService,  new google.maps.LatLng(car.lat,car.lon), tuberlin,1);
                    myinterval = setInterval(showCar, 1000);
                    startSimulation();}
                    ,5000);
            } else {
                powerstate = car.soc * car.capacity;
                //alert("powerPD" +car.powerPD);
                consumption = 0.001 * car.powerPD;
                //alert(newLocation);
                if (powerstate - consumption > 0) {
                    //alert("powerstate"+powerstate);
                    //alert("consumption"+consumption);
                    // available energy - needed energy
                    powerupdate = powerstate - consumption;
                    // percentage battery left (new state of charge)
                    soc_update = powerupdate / car.capacity;

                    car.soc = soc_update;
                    car.lat = locationsToChargingStation[0].lat();
                    car.lon = locationsToChargingStation[0].lng();
                    //alert("car location"+car.lat,car.lon);
                    carMarker.setPosition(locationsToChargingStation[0]);
                    map.setCenter(carMarker.position);
                    showCar();
                    //alert ("soc"+soc_update)
                   locationsToChargingStation.shift();
                }
            }
        },
        100);
}

function showGrid(selectedGrid){
    $("grid").removeAttr("hidden");
    //removeRouteMarker();
    $('#gridName').val( selectedGrid.name) ;
    $('#p_chargingStation').val(selectedGrid.p_charging_station);
    $('#price').val(selectedGrid.price);
    $('#alpha').attr('src', 'static/images/energy mix for grid'+selectedGrid.id+'.png');

    //initCarMarkers();
    //initStationMarkers();

    /*stationMarkers.forEach(function (marker)
    {
        stationId = marker.title.split("\n")[0];
        if (stationId == 'station'+gridList[i].id)
        {
            marker.setIcon(selectedStationIcon);
            map.setCenter(marker.position);
        }
    });
*/

}

function calculateAndDisplayRoute(directionsDisplay, directionsService, start, end, isDestination ) {
    directionsDisplay.setMap(map);
    removeRouteMarker();
    directionsService.route({
        origin: start,
        destination:end,
        travelMode: 'DRIVING'
    }, function(response, status) {
        if (status === 'OK') {
            polyline.setPath([]);
            directionsDisplay.setDirections(response);
            let legs = response.routes[0].legs;
            if(isDestination)
                routeMarkers.push(addMarker( "destination",legs[0].end_location,destinationIcon));
            for (var i = 0; i < legs.length; i++) {
                var steps = legs[i].steps;
                for (var j = 0; j < steps.length; j++) {
                    var nextSegment = steps[j].path;
                    for (var k = 0; k < nextSegment.length; k++) {
                        polyline.getPath().push(nextSegment[k]);
                    }
                }
            }
            locationsToDistination = [];
            locationsToChargingStation=[];
            var totalDist = polyline.Distance();
            for (var l = 0; l <totalDist ; l += 14) {
                if (isDestination)
                    locationsToDistination.push(polyline.GetPointAtDistance(l));
                else
                    locationsToChargingStation.push(polyline.GetPointAtDistance(l));

            }
        }
        else window.alert('Directions request failed due to ' + status);
    });
}

function getBatteryIcon(soc){
    let src;
    if ( soc <= 20 && soc > 0 )
        src = 'static/images/20.png' ;
    else if ( soc <= 40 && soc > 20 )
        src = 'static/images/40.png' ;
    else if (soc < 60 && soc > 40  )
        src = 'static/images/60.png' ;
    else if (soc <= 80 && soc >= 60 )
        src = 'static/images/80.png' ;
    else if ( soc <= 100 && soc > 80  )
        src = 'static/images/100.png' ;
    return src ;
}

function getBatteryIcon2(soc){
    let src;
    if  (soc === 0)
        src = 'static/images/B_0%.png' ;
    else if ( soc <= 10 && soc > 0 )
        src = 'static/images/B_10%.png' ;
    else if ( soc <= 20 && soc > 10 )
        src = 'static/images/B_20%.png' ;
    else if (soc < 30 && soc > 20  )
        src = 'static/images/B_30%.png' ;
    else if (soc <= 40 && soc >= 30 )
        src = 'static/images/B_40%.png' ;
    else if ( soc <= 50 && soc > 40  )
        src = 'static/images/B_50%.png' ;
    else if ( soc <= 60 && soc > 50  )
        src = 'static/images/B_50%.png' ;
    else if ( soc <= 70 && soc > 60  )
        src = 'static/images/B_60%.png' ;
    else if ( soc <= 80 && soc > 70  )
        src = 'static/images/B_70%.png' ;
    else if ( soc <= 90 && soc > 80  )
        src = 'static/images/B_80%.png' ;
    else if ( soc <= 100 && soc > 90  )
        src = 'static/images/B_90%.png' ;
    else if ( soc === 100  )
        src = 'static/images/B_100%.png' ;
    return src ;
}

// This function displays the route in driving mode between origin and fixed goal tu berlin
// https://developers.google.com/maps/documentation/javascript/examples/directions-simple
function calculateDisplayRoutes(lat, long) {
     let origin = new google.maps.LatLng(lat, long);
     let routePoints = [];
     let pointDuration = [];
     directionsService.route({
          origin: origin,
          destination: tuberlin,
          travelMode: 'DRIVING'
        }, function(response, status) {
          if (status === 'OK') {
            // comment following line to not display route
            directionsDisplay.setDirections(response);
 		    for(var i=0; i<response.routes[0].legs[0].steps.length; i++) {
                // get all turning points of route
                routePoints.push(response.routes[0].legs[0].steps[i].end_location);
        // time in minutes for each part of route
                pointDuration.push(response.routes[0].legs[0].steps[i].duration.value/60);
            }
            calculateNewCarPosition(routePoints.unshift(origin), pointDuration);
          } else {
            window.alert('Directions request failed due to ' + status);
          }
     });
}

function calculateNewCarPosition(routePoints, pointDuration) {
    alert("point routePoints------",routePoints);
    for (let i=0; i<routePoints.length-1; i++) {
        let start, end, distanceLat, distanceLong;
        start = routePoints[i];
        end = routePoints[i+1];
        distanceLat = start[0]-end[0];
        distanceLong = start[1]-end[1];
        increaseLat = distanceLat/Math.floor(pointDuration[i]);
        increaseLong = distanceLong/Math.floor(pointDuration[i]);
        for (let j=0; j<Math.floor(pointDuration[i]); j++) {
            // increase start coordinate towards end coordinate
            gpsCoords.push([routePoints[i][0]+increaseLat, routePoints[i][1]+increaseLong]);
        }
        if (Math.floor(pointDuration[i]) < pointDuration[i]) {
            // perform last step of routePart
            gpsCoords.push(end);
        }
    }
    for (let i=0; i<gpsCoords.length; i++) {
        timer = setInterval(updateGPS, 1000, gpsCoords[i]);
        alert("gps coordinates -------",gpsCoords);
    }
}

function updateGPS() {
            // update car icon to gpsCoords i or do a smooth transition animation ?
            // make coords available to optimization
            newcoords = gpsCoords[i];
            alert(newcoords);
            newIconCoords = new google.maps.LatLng(gpsCoords[i][0], gpsCoords[i][1]);
}

