/** INIT ICONS **/
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

/** INIT GLOBAL VARS **/
let map;
let autoDriveTimer;
let newTimer;
let directionsDisplay;
let timer;
var myinterval;
let myinterval2;
let routeMarkers=[];
let carMarkers=[];
let carMarker;
let stationMarkers=[];
let locationsToDistination =[];
let locationsToChargingStation =[];
let routePoints = [];
let pointDuration = [];
let gpsCoords = [];
let distances = [];
let directionsService = new google.maps.DirectionsService;
let destinationIcon = 'https://chart.googleapis.com/chart?' + 'chst=d_map_pin_letter&chld=D|FF0000|000000';
let flag = true;
let endPoints = [new google.maps.LatLng(52.512230, 13.327135), new google.maps.LatLng(52.5217707,13.3295431),
new google.maps.LatLng(52.5466252,13.3505417), new google.maps.LatLng(52.5314232,13.4353983), 
new google.maps.LatLng(52.4851232,13.3607773), new google.maps.LatLng(52.5106902,13.4290883),
new google.maps.LatLng(52.5273782,13.4044253), new google.maps.LatLng(52.4884422,13.3019483)];
var randPoint = new google.maps.LatLng(52.512230, 13.327135);
var polyline = new google.maps.Polyline({
    path: [],
    strokeColor: '#FF0000',
    strokeWeight: 3
});

/** DOCUMENT READY -- START SIMULATION **/
$(document).ready(function() {
    // remove remains from previous runs
    locationsToDistination = [];
    // init Map & load Grids
    initMap();
    loadGrids(gridList);
    // display routes to random goal 
    randPoint = endPoints[Math.floor(Math.random() *  endPoints.length-1)];
    calculateAndDisplayRoute(directionsDisplay, directionsService,  new google.maps.LatLng(car.lat,car.lon), endPoints[Math.floor(Math.random() *  endPoints.length-1)], 1);
    // remove car marker to clear map
    removeCarsMarker();
    // add car marker, set to car.lat + car.long and start Interval to update car position
    carMarker= addMarker('car: '+ car.id +'\nsoc: '+car.soc,new google.maps.LatLng(car.lat,car.lon),selectedCarIcon);
    carMarkers.push(carMarker);

    myinterval = setInterval(showCar, 500);
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

    carMarker.setPosition(new google.maps.LatLng(car.lat,car.lon));
    map.setCenter(carMarker.position);
    
    if(locationsToDistination.length > 0 && flag) {
        startSimulation();
        alert("simulation is starting")
        }
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
        300);
}

function moveCar(newLocation){
    powerstate = car.soc*car.capacity;
    consumption = 0.001*car.powerPD;

    if (powerstate-consumption > 0){
        // available energy - needed energy
        powerupdate = powerstate-consumption;
        // percentage battery left (new state of charge)
        soc_update = powerupdate/car.capacity;

        if (soc_update <= 0) {
            alert("Battery below 1%")
        }
        if (soc_update <= 0.2) {
            alert (" POWER IS LOW \n Searching for a charging station...");
            clearInterval(autoDriveTimer);
            get_grids();

            var reachable_grids = [];
            var length = 0;
            var lat = parseFloat(locationsToDistination[0].lat());
            var lng = parseFloat(locationsToDistination[0].lng());
            var origin = new google.maps.LatLng(lat, lng);

            var p2 = new Promise(function(resolve, reject) {
                $.ajax({
                 type: "POST",
                 url: "/load_grids",
                 data: JSON.stringify(car),
                 success: function(data){
                    resolve(data);
                },dataType: 'json'
                });
            }).then(function (data) {
                var grids = [];
                
                for (var i=0; i<data.length; i++) {
                    var lat_g = parseFloat(data[i].lat);
                    var lng_g = parseFloat(data[i].lon);
                    var end = new google.maps.LatLng(lat_g, lng_g);
                    var length = 0; 
                    
                    grids[i] = new Promise(function(resolve, reject){
                    
                       directionsService.route({
                                origin: origin,
                                destination: end,
                                travelMode: 'DRIVING'
                            }, function(response, status) {
                                if (status === 'OK') {
                                    for(var i=0; i<response.routes[0].legs.length; i++) {
                                        length += response.routes[0].legs[i].distance.value; 
                                    }
                                    resolve(length);
                                    flag = true; 
                                } else {
                                    console.log(status);
                                }
                            });

                    }); 

                }
                Promise.all(grids).then(function(datas) {
                    console.log("OUTPUT");
                    console.log(datas);
                    console.log(datas.length);
                    console.log(datas[0]);
                });
                });  
        
            /**p.then(function(distances) {
                console.log("searching for reachable grids...");
                console.log("RESULTS");
                console.log(distances);
                console.log(distances.length);
                reachable_grids = reachableGrids(car, distances);
            });**/


            // POSSIBLY NOT NEEDED, UPDATE REACHABILITY BELOW BASED ON REAL DRIVING DISTANCE,
            
                         /**$.ajax({
                 type: "POST",
                 url: "/postCar_getGrid",
                 data: JSON.stringify(car),
                 success: function(data){
                     alert("The optimal grid is grid"+data.id);
                     //showGrid(data.id);
                     let gridLocation = new google.maps.LatLng(data.lat, data.lon);
                     //alert(data);
                     $("#grid").removeAttr("hidden");
                     //showGrid(data);
                     alert("gridLocation..." + gridLocation);
                     calculateAndDisplayRoute(directionsDisplay, directionsService,newLocation,gridLocation,0);
                     driveToCharginStation();
        }
        ,dataType: 'json'
    });**/
        } else {
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

function updateReachableGrids(reachable_grids, callback) {

}

function calcDist(grid, callback) {



    var lengths = [];


}     

// TO DO : REFACTOR
function reachableGrids(car, distances) {
    var final_grids = [];
    var total_charge;
    console.log("LENGTH "+ distances);
    // dist per step: 14m // decay per step: consumption = 0.5*car.powerPD
    // line 371 & 230 / line 430
    for (var i; i<distances.length; i++) {
        /** calculate total energy to grid considering distance
         divide the total distance to goal by the length of one decay step (14m), 
         multiply the result by the battery consumption per step **/
        var c_to_grid = car.powerkm*0.001 * (distances[i]/14); 
        console.log(c_to_grid +" charge to grid");
        console.log(car.state_of_charge + " soc");
        // consider only reachable grids
        if (c_to_grid <= (car.state_of_charge*car.capacity)){
            // add the energy needed to reach grid to the total deficit
            total_charge = c_to_grid+(car.capacity-(car.state_of_charge*car.capacity));
            // location, total energy needed at grid location, distance, price
            final_grids.append(grids[x]);
            // TO DO: UPDATE TOTAL CHARGE FOR OPTIMIZATION iN DATABASE OR SOMETHING SIMILAR

        }
    }
    console.log("Grids within reach: ", final_grids);
    return final_grids;
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

                    calculateAndDisplayRoute(directionsDisplay, directionsService,  new google.maps.LatLng(car.lat,car.lon), randPoint,1);
                    myinterval = setInterval(showCar, 500);
                    startSimulation();}
                    ,5000);
            } else {
                powerstate = car.soc * car.capacity;
                //alert("powerPD" +car.powerPD);
                consumption = 0.001 * car.powerPD;
                //alert(newLocation);
                if (powerstate - consumption > 0) {
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
        300);
}

function showGrid(selectedGrid){
    $("grid").removeAttr("hidden");
    //removeRouteMarker();
    $('#gridName').val( selectedGrid.name) ;
    $('#p_chargingStation').val(selectedGrid.p_charging_station);
    $('#price').val(selectedGrid.price);
    $('#alpha').attr('src', 'static/images/energy mix for grid'+selectedGrid.id+'.png');
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
        timer = setInterval(updateGPS, 500, gpsCoords[i]);
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

