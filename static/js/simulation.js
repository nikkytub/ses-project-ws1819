let map;
let carIcon = {
    url: "static/images/car.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedCarIcon = {
    url: "static/images/selected-car.png",
    scaledSize: new google.maps.Size(40, 40)
};

let stationIcon = {
    url: "static/images/station.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedStationIcon = {
    url: "static/images/selected-station.png",
    scaledSize: new google.maps.Size(40, 40)
};
let autoDriveTimer;
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
let locations2=[];
let locations =[];
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
    //showCar();
    calculateAndDisplayRoute(directionsDisplay, directionsService,  new google.maps.LatLng(car.lat,car.lon), tuberlin);
    removeCarsMarker();
    carMarker= addMarker('car'+ car.id +'\nsoc: '+car.soc
            , new google.maps.LatLng(car.lat,car.lon),selectedCarIcon)
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

function loadGrids(gridList){
    for (let i = 0 ;i<gridList.length;i++)
    {

        let stationMarker =addMarker('station'+ gridList[i].id +'\nprice: '+gridList[i].price
            +'\ncapacity: '+gridList[i].capacity, new google.maps.LatLng(gridList[i].lat,gridList[i].lon),stationIcon);
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

function removeCarsMarker(){
    carMarkers.map(function(marker){
        marker.setMap(null);
    } );
    carMarkers = []
}


function showCar(){
    clearInterval(myinterval);
    $('#speedInput').val(car.speed);
    if(car.mode == "eco_mode")
        $('#eco').bootstrapToggle('on');
    else if (car.mode == "costSaving_mode")
        $('#cost').bootstrapToggle('on');
    else if (car.mode == "chargingtime_mode")
        $('#time').bootstrapToggle('on');
    $('#battery').attr('src', getBatteryIcon(car.soc * 100));
    $('#b1soc').text(car.soc * 100 + '%');
    //removeCarsMarker();
    //carMarker= addMarker('car'+ car.id +'\nsoc: '+car.soc
     //       , new google.maps.LatLng(car.lat,car.lon),selectedCarIcon)
    //carMarkers.push(carMarker);
    carMarker.setPosition(new google.maps.LatLng(car.lat,car.lon));
    map.setCenter(carMarker.position);
    if(locations.length > 0 && flag)
    {startSimulation();
    alert("simulation is starting")}
    }


function startSimulation(){
    flag = false;

    autoDriveTimer = setInterval(function () {
            // stop the timer if the route is finished
            if (locations.length === 0) {
                clearInterval(autoDriveTimer);
            } else {

                if (car.soc > 0.2) {
                    // move marker to the next position (always the first in the array)

                    moveCar(locations[0]);
                    // remove the processed position
                    locations.shift();
                }
               else {

                }
            }
        },
        1000);


    //for (i = 5; i<locations.length; i++){
    //    window.setTimeout(function() {moveCar(locations[i]);
    //    }, 5000);

    //}
}

function moveCar(newLocation){

    powerstate = car.soc*car.capacity;
    //alert("powerPD" +car.powerPD);
    consumption = 0.9*car.powerPD;
    //alert(newLocation);
    if (powerstate-consumption > 0){
        //alert("powerstate"+powerstate);
        //alert("consumption"+consumption);
        // available energy - needed energy
        powerupdate = powerstate-consumption;
        // percentage battery left (new state of charge)
        soc_update = Math.floor((powerupdate/car.capacity) *100 )/100;
        //alert ("soc"+soc_update);
        if (soc_update <= 0) {
            //alert(soc_update);
            //alert("Battery below 1%.. Randomly resetting SOC to ", soc_update)
        }
        if (soc_update <= 0.2) {
            alert (" POWER IS LOW \n Searching for a charging station...");
            clearInterval(autoDriveTimer);
             $.ajax({
                 type: "POST",
                 url: "/postCar_getGrid",
                 data: JSON.stringify(car),
                 success: function(data){
                     showGrid(data.id);
                     let gridLocation = new google.maps.LatLng(data.lat, data.lon);
                     //alert("gridLocation,,,," + gridLocation);
                     calculateAndDisplayRoute(directionsDisplay, directionsService,carLocation,gridLocation );
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
            //alert("car location"+car.lat,car.lon);
            carMarker.setPosition(newLocation);
            map.setCenter(carMarker.position);
        }
        showCar();
    }

}


function showGrid(selectedGrid){
    directionsDisplay.setMap(null);
    removeRouteMarker();
    $('#stations option').each(function(){
            let $this = $(this); // cache this jQuery object to avoid overhead

            if ($this.val() == selectedGrid) { // if this option's value is equal to our value
             $this.prop('selected', true); // select this option
                return false; // break the loop, no need to look further
            }
        });
    for (let i =0 ; i < gridList.length; i++){
        if (gridList[i].id == selectedGrid)
        {
            $("#stations").selectpicker("refresh");
            $('#stationLocation').val( "("+gridList[i].lat + ","+gridList[i].lon+")") ;
            $('#capacity').val(gridList[i].capacity);
            $('#price').val(gridList[i].price);

            initCarMarkers();
            initStationMarkers();

            stationMarkers.forEach(function (marker)
            {
                stationId = marker.title.split("\n")[0];
                if (stationId == 'station'+gridList[i].id)
                {
                    marker.setIcon(selectedStationIcon);
                    map.setCenter(marker.position);
                }
            });
        }
    }
}

function calculateAndDisplayRoute(directionsDisplay, directionsService, start, end ) {
    locations2= [];
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
            var totalDist = polyline.Distance();
            for (var l = 0; l <totalDist ; l += 14) {
                    locations.push(polyline.GetPointAtDistance(l));
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

