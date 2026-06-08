import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../bloc/safety_bloc.dart';
import '../bloc/safety_state.dart';

class SafetyScreen extends StatelessWidget {
  const SafetyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Safety")),

      body: BlocBuilder<SafetyBloc, SafetyState>(
        builder: (context, state) {
          if (state is SafetyLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          if (state is SafetyLoaded) {
            return ListView.builder(
              itemCount: state.instructions.length,

              itemBuilder: (context, index) {
                final item = state.instructions[index];

                return Card(
                  margin: const EdgeInsets.all(12),

                  child: Padding(
                    padding: const EdgeInsets.all(16),

                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,

                      children: [
                        Text(
                          item.type,

                          style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        const SizedBox(height: 16),

                        Text("Before: ${item.before}"),

                        const SizedBox(height: 8),

                        Text("During: ${item.during}"),

                        const SizedBox(height: 8),

                        Text("After: ${item.after}"),
                      ],
                    ),
                  ),
                );
              },
            );
          }

          if (state is SafetyError) {
            return Center(child: Text(state.message));
          }

          return const SizedBox();
        },
      ),
    );
  }
}
