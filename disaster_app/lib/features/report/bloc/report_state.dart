abstract class ReportState {}

class ReportInitial extends ReportState {}

class ReportLoading extends ReportState {}

class ReportSuccess extends ReportState {}

class ReportError extends ReportState {
  final String message;

  ReportError(this.message);
}
