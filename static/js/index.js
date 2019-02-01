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
let routeMarkers=[];
let carMarkers=[];
let stationMarkers=[];
let directionsDisplay ;
let directionsService = new google.maps.DirectionsService;
let destinationIcon = 'https://chart.googleapis.com/chart?' + 'chst=d_map_pin_letter&chld=D|FF0000|000000';


$(document).ready(function() {
    initMap();
    $("#cars").html("");
    $("#stations").html("");
    loadCars(carList);
    loadGrids(gridList);
    carMarkers.forEach(function (marker){
        marker.addEventListener('click', function () {
            map.setZoom(8);
            map.setCenter(marker.getPosition());
        });

    } )
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
        $("#stations").append( '<option  value='+gridList[i].id+'>station' + gridList[i].id + '</option>' );
        let stationMarker =addMarker('station'+ gridList[i].id +'\nprice: '+gridList[i].price
            +'\ncapacity: '+gridList[i].capacity, new google.maps.LatLng(gridList[i].lat,gridList[i].lon),stationIcon);
          stationMarker.addListener('click', function() {
              showGrid(gridList[i].id);
          });
        stationMarkers.push(stationMarker);
    }
    $("#stations").selectpicker("refresh");

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

$("#searchButton").click(function(){
    let data =  {};
    let carLocation ;
    data['id'] = $("#cars").find("option:selected").val();
    data['mode'] = $("#mode").find("option:selected").val();
    //alert(data['mode']);
    for (let i =0 ; i < carList.length ; i++)
    {       //alert(carList[i].id +'========'+ data['id']);
        if (carList[i].id == data['id']){
             carLocation = new google.maps.LatLng(carList[i].lat,carList[i].lon);
             //alert("carLocation,,,," + carLocation);
        }}
    $.ajax({
    type: "POST",
    url: "/postCar",
    data: JSON.stringify(data),
    success: function(data){
        showGrid(data.id);
        let gridLocation = new google.maps.LatLng(data.lat, data.lon);
        //alert("gridLocation,,,," + gridLocation);
        calculateAndDisplayRoute(directionsDisplay, directionsService,carLocation,gridLocation );
        }
        ,dataType: 'json'
    });
    //var gridList = JSON.parse('{{ optimal_grid | tojson | safe}}');
    //var optimal_grid = JSON.parse('{{ optimal_grid | tojson | safe}}');

    //alert(gridList);

});
$("#simulationButton").click(function() {
    window.location.assign("/simulation?car="+ $("#cars").find("option:selected").val() );
});

function addMarker(title,position,icon) {
    return new google.maps.Marker({
        title:title,
        position: position,
        map: map,
        icon: icon
    });
}

function showCar(selectedCar){
    directionsDisplay.setMap(null);
    removeRouteMarker();
    $('#cars option').each(function(){
            let $this = $(this); // cache this jQuery object to avoid overhead

            if ($this.val() == selectedCar) { // if this option's value is equal to our value
             $this.prop('selected', true); // select this option
                return false; // break the loop, no need to look further
            }
        });

    for (let i =0 ; i < carList.length ; i++)
    {
        if (carList[i].id == selectedCar)
        {
            $("#cars").selectpicker("refresh");
            $('#location').val("(" + carList[i].lat + "," + carList[i].lon + ")");
            $('#speedInput').val(carList[i].speed);
            $('#mode option').each(function(){
            let $this = $(this); // cache this jQuery object to avoid overhead

            if ($this.val() == carList[i].mode) { // if this option's value is equal to our value
             $this.prop('selected', true); // select this option
                return false; // break the loop, no need to look further
            }
        });
            $("#mode").selectpicker("refresh");
            $('#battery').attr('src', getBatteryIcon(carList[i].soc * 100));
            $('#b1soc').text(carList[i].soc * 100 + '%');
            if (carList[i].soc <= 0.2)
            {
                $('#mode').removeAttr('disabled');
                $('#searchButton').removeAttr('hidden');
            }
            initCarMarkers();
            initStationMarkers();

            carMarkers.forEach(function (marker) {
                let carId = marker.title.split("\n")[0];
                if (carId == 'car'+carList[i].id)
                {
                    marker.setIcon(selectedCarIcon);
                    map.setCenter(marker.position);
                }
            });
        }
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

function initCarMarkers(){
    carMarkers.map(function(marker){
        marker.setIcon(carIcon);
    } );
}

function initStationMarkers(){
    stationMarkers.map(function(marker){
        marker.setIcon(stationIcon);
    } );
}

function removeRouteMarker(){
    routeMarkers.map(function(marker){
        marker.setMap(null);
    } );
    routeMarkers = []
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

function calculateAndDisplayRoute(directionsDisplay, directionsService, start, end ) {
    directionsDisplay.setMap(map);
    removeRouteMarker();
    directionsService.route({
        origin: start,
        destination:end,
        travelMode: 'DRIVING'
    }, function(response, status) {
        if (status === 'OK') {
            directionsDisplay.setDirections(response);
            let legs = response.routes[0].legs;

            routeMarkers.push(addMarker( "destination",legs[0].end_location,destinationIcon));
        }
        else window.alert('Directions request failed due to ' + status);
    });
}
