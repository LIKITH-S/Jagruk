import '../models/safety_model.dart';

abstract class SafetyState {}

class SafetyLoading extends SafetyState {}

class SafetyLoaded extends SafetyState {
  final List<SafetyModel> instructions;

  SafetyLoaded(this.instructions);
}

class SafetyError extends SafetyState {
  final String message;

  SafetyError(this.message);
}
