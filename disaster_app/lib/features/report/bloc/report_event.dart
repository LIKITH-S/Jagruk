abstract class ReportEvent {}

class SubmitReport extends ReportEvent {
  final String description;

  final String disasterType;

  final double lat;
  final double lon;

  SubmitReport({
    required this.description,

    required this.disasterType,

    required this.lat,
    required this.lon,
  });
}
