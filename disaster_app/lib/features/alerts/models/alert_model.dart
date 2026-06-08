class AlertLocation {
  final double lat;
  final double lon;

  AlertLocation({required this.lat, required this.lon});

  factory AlertLocation.fromJson(Map<String, dynamic> json) {
    return AlertLocation(lat: json['lat'], lon: json['lon']);
  }
}

class Alert {
  final String id;
  final String type;
  final String severity;

  final AlertLocation location;

  Alert({
    required this.id,
    required this.type,
    required this.severity,
    required this.location,
  });

  factory Alert.fromJson(Map<String, dynamic> json) {
    return Alert(
      id: json['id'],
      type: json['type'],
      severity: json['severity'],

      location: AlertLocation.fromJson(json['location']),
    );
  }
}
