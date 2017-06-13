function get_url_var() {
	var urlParams;
	(window.onpopstate = function () {
		var match,
			pl     = /\+/g,  // Regex for replacing addition symbol with a space
			search = /([^&=]+)=?([^&]*)/g,
			decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
			query  = window.location.search.substring(1);

		urlParams = {};
		while (match = search.exec(query))
			urlParams[decode(match[1])] = decode(match[2]);
	})();
	return urlParams;
}

//get home path
pathArray = location.href.split( '/' );
protocol = pathArray[0];
host = pathArray[2];
var domain_name = protocol + '//' + host;
var url_params = get_url_var();
var filename = url_params['search'];

$("#form").attr("action", domain_name + ":5000/upload");

if (filename!=null){
	var url = domain_name + ':5000/recommend?img=' + filename + '&num=9';
	$.ajax({
		url: url,
		dataType: "jsonp",
		jsonp: 'callback',
		jsonpCallback: 'jsonp_callback',
		success: function (data) {
			var status = data['status'];
			if (status == 0) {
				var category = data['category'];
				var products = data['recommendations'];
				$("#result").html("");
				for (var row=0; row<3; row++) {
					$("#result").append('<div class="w3-row-padding">');
					for (var col=0; col<3; col++){
						var asin = products[String(row*3+col)]['asin'];
						var imUrl = products[String(row*3+col)]['imUrl'];
						var page = products[String(row*3+col)]['page'];
						$("#result").append('<div class="w3-third w3-container w3-margin-bottom">'+
							'<img src="'+imUrl+'" alt="Norway" style="width:100%" class="w3-hover-opacity">'+
	    					'<div class="w3-container w3-white">'+
					        '<p><b>Category: '+category+'</b></p>'+
					        '<p><a href="'+page+'">Buy it here</a></p>'+
						    '</div>'+'</div>');
					}
					$("#result").append('</div>')
				}
			}
		}
	});
};

$(document).ready(function() {
    $('#result').show();
});
