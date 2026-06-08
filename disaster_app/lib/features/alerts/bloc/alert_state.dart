import '../models/alert_model.dart';

abstract class AlertState {}

class AlertLoading extends AlertState {}

class AlertLoaded extends AlertState {
  final List<Alert> alerts;

  AlertLoaded(this.alerts);
}

class AlertError extends AlertState {
  final String message;

  AlertError(this.message);
}
