class Alert {
  final String id;
  final String type;
  final String severity;
  final double lat;
  final double lon;

  Alert({
    required this.id,
    required this.type,
    required this.severity,
    required this.lat,
    required this.lon,
  });

  factory Alert.fromJson(Map<String, dynamic> json) {
    return Alert(
      id: json['id'],
      type: json['type'],
      severity: json['severity'],
      lat: json['location']['lat'],
      lon: json['location']['lon'],
    );
  }
}
