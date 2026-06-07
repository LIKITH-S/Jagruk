import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'core/services/mock_alert_service.dart';
import 'features/alerts/bloc/alert_bloc.dart';
import 'features/alerts/screens/alerts_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: BlocProvider(
        create: (_) => AlertBloc(MockAlertService())..add(LoadAlerts()),
        child: const AlertsScreen(),
      ),
    );
  }
}
